
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.evaluation import evaluate_policy




SYMBOL = "BTC/USDT"
EXCHANGE = "binance"
TIMEFRAME = "5m"

WINDOW_SIZE = 48          # lookback of ~4 hours at 5m candles  
EPISODE_LENGTH = 500      # one episode = 500 steps (~1.7 days at 5m candles)
TOTAL_TIMESTEPS = 1_000_000 # much more training

INITIAL_BALANCE = 10_000.0

BUY_THRESHOLD = 0.02      # min. predicted prob. to open buy  
SELL_THRESHOLD = -0.02     # min. predicted prob. to open sell  
MAX_TRADE_FRACTION = 0.25 # max 25% of balance per trade

TRANSACTION_COST_PCT = 0.0005   # 0.05% realistic fee  
SLIPPAGE_PCT = 0.0005           # 0.05% slippage  
INVALID_ACTION_PENALTY = -1.0   # stronger penalty for doing nothing wrong

# AI / PPO settings
N_STEPS = 2048            # rollout length before each update
BATCH_SIZE = 64           # batch size for training
ENTROPY_COEFFICIENT = 0.01 # encourages exploration (default PPO ~0.0–0.01)
LEARNING_RATE = 3e-4      # stable-baselines default, works well for PPO



MODEL_DIR = "./models_crypto/"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "crypto.zip")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# -----------------------
# Fetch OHLCV data
# -----------------------
print("Fetching OHLCV data from", EXCHANGE, SYMBOL, TIMEFRAME)
exchange = getattr(ccxt, EXCHANGE)()
since = exchange.parse8601("2025-04-01T00:00:00Z")

all_bars = []
while True:
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000)
    if not bars:
        break
    all_bars += bars
    since = bars[-1][0] + 1
    if len(bars) < 1000:
        break

df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.reset_index(drop=True)

prices = df["close"].to_numpy(dtype=np.float32)
volumes = df["volume"].to_numpy(dtype=np.float32)
N = len(prices)
print(f"Loaded {N} bars for {SYMBOL}")

if N < WINDOW_SIZE + EPISODE_LENGTH + 10:
    raise RuntimeError("Not enough bars loaded for chosen WINDOW_SIZE + EPISODE_LENGTH.")

split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_test = df.iloc[split_idx:].reset_index(drop=True)

train_prices = df_train["close"].to_numpy(dtype=np.float32)
train_vols = df_train["volume"].to_numpy(dtype=np.float32)

test_prices = df_test["close"].to_numpy(dtype=np.float32)
test_vols = df_test["volume"].to_numpy(dtype=np.float32)


class TradingEnvContinuous(gym.Env):
    """
    Continuous-action trading environment.
    Action is a single scalar in [-1, 1] representing the model's predicted signed return/confidence:
      - positive -> expect price to go up (long)
      - negative -> expect price to go down (short / sell)
    The env maps that prediction into a proportional buy/sell:
      - pred > buy_threshold : buy fraction = min(max_trade_fraction, pred)
      - pred < sell_threshold: sell fraction = min(max_trade_fraction, -pred)
      - otherwise: hold

    Use with 5-min candles (or any timeframe) — obs and reward unchanged from your original env.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, timestamps=None, volumes=None, window_size=48, initial_balance=10000.0,
                 transaction_cost_pct=0.0, slippage_pct=0.0, episode_length=200,
                 buy_threshold=0.05, sell_threshold=-0.05, max_trade_fraction=0.5):
        super().__init__()

        # Core data
        self.prices = np.array(prices, dtype=np.float32).ravel()
        if timestamps is None:
            self.timestamps = np.arange(len(self.prices))
        else:
            self.timestamps = np.array(timestamps)
        self.volumes = np.array(volumes, dtype=np.float32).ravel() if volumes is not None else np.ones_like(self.prices)

        # config
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.slippage_pct = float(slippage_pct)
        self.episode_length = int(episode_length)

        # Continuous-action specific params
        # prediction thresholds: predictions inside (sell_threshold, buy_threshold) => HOLD
        self.buy_threshold = float(buy_threshold)   # e.g. 0.05 (5% confidence)
        self.sell_threshold = float(sell_threshold) # e.g. -0.05
        # maximum fraction of current balance to use for a single buy / maximum fraction of shares to sell
        self.max_trade_fraction = float(max_trade_fraction)  # e.g. 0.5 => at most 50% balance per buy

        # features / spaces (keep same num_features to match your obs builder)
        self.num_features = 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size * self.num_features,), dtype=np.float32)
        # single continuous scalar in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # env state will be set in reset()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.prices) - self.episode_length - 1
        if max_start <= self.window_size:
            self.start_step = self.window_size
        else:
            self.start_step = int(np.random.randint(self.window_size, max_start))

        self.current_step = int(self.start_step)
        self.steps_left = int(self.episode_length)

        # portfolio state
        self.balance = float(self.initial_balance)
        self.shares = 0.0       # fractional shares allowed
        self.prev_total = float(self.initial_balance)
        self.peak_balance = float(self.initial_balance)

        # logs
        self.trades = []        # (typ, step, price, qty_float)
        self.history = []       # list of dicts (step,timestamp,price,balance,shares,portfolio_value)

        self.done = False
        
        self.inventory_cost = 0.0    # total USD spent to acquire current shares (includes entry fees)
        self.prev_action = 0

        return self._get_obs(), {}

    def _get_obs(self):
        start = int(self.current_step - self.window_size)
        if start < 0:
            start = 0
        window_prices = self.prices[start:self.current_step]
        if len(window_prices) < self.window_size:
            pad_len = self.window_size - len(window_prices)
            pad_val = window_prices[0] if len(window_prices) > 0 else self.prices[self.current_step]
            pad = np.full(pad_len, pad_val, dtype=window_prices.dtype)
            window_prices = np.concatenate([pad, window_prices])

        obs = np.zeros((self.window_size, self.num_features), dtype=np.float32)
        prev = np.concatenate([[window_prices[0]], window_prices[:-1]])
        # returns / price change
        obs[:, 0] = (window_prices - prev) / (prev + 1e-8)

        # other features left as zeros unless you compute them (you can compute RSI/EMA externally and pass in)
        # Keep vol normalized in last column like before
        max_vol = np.max(self.volumes) if np.max(self.volumes) > 0 else 1.0
        vol_w = self.volumes[start:self.current_step]
        vol_w = np.pad(vol_w, (self.window_size - len(vol_w), 0), 'constant') if len(vol_w) < self.window_size else vol_w
        obs[:, 6] = np.nan_to_num(vol_w / max_vol, nan=0.0)

        flat = obs.flatten()
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return flat

    def step(self, action):
        """
        Updated step() with:
        - reward normalization by prev_total for denser signal
        - safety guards (inventory_cost never negative)
        - turnover normalized by prev_total
        - repeated 'sell while flat' treated as hold (no penalty)
        """

        if self.done:
            raise RuntimeError("step() called after done; call reset().")

        # ---------- Reward-shaping hyperparams (tune these) ----------
        REALIZED_WEIGHT = 1.0
        UNREALIZED_WEIGHT = 0.15
        TURNOVER_PENALTY_COEF = 0.5
        CHURN_PENALTY = 0.0005
        DRAWDOWN_PENALTY_COEF = 0.2
        MIN_TRADE_PENALTY = 0.0
        # -------------------------------------------------------------

        eps = 1e-12

        # coerce action scalar
        if isinstance(action, (list, tuple, np.ndarray)):
            pred = float(np.asarray(action).ravel()[0])
        else:
            pred = float(action)

        # clip prediction to [-1,1]
        pred = float(np.clip(pred, self.action_space.low[0], self.action_space.high[0]))

        price = float(self.prices[self.current_step])
        realized_profit = 0.0
        turnover_fraction = 0.0
        trade_count_this_step = 0

        prev_shares = float(self.shares)
        prev_inventory_cost = float(getattr(self, "inventory_cost", 0.0))
        prev_total = float(getattr(self, "prev_total", self.initial_balance))

        # ---------- Interpret prediction and execute proportional trade ----------
        if pred > self.buy_threshold:
            # BUY
            frac = min(self.max_trade_fraction, max(0.0, pred))
            usd_alloc = self.balance * frac
            exec_price = price * (1.0 + self.slippage_pct)
            cost_multiplier = 1.0 + self.transaction_cost_pct
            qty = usd_alloc / (exec_price * cost_multiplier) if exec_price > 0 else 0.0

            if qty > 1e-12:
                total_cost = qty * exec_price * cost_multiplier
                qty_executed = qty
                if total_cost > self.balance:
                    qty_executed = (self.balance / (exec_price * cost_multiplier))
                    total_cost = qty_executed * exec_price * cost_multiplier

                self.balance -= total_cost
                self.inventory_cost += total_cost    # include fees in cost basis
                # safety: never go negative
                if self.inventory_cost < 0:
                    self.inventory_cost = 0.0

                self.shares += qty_executed

                turnover_fraction = (total_cost) / (prev_total + eps)
                trade_count_this_step += 1
            else:
                # tiny bookkeeping negative to reflect failed buy (very small)
                realized_profit -= 1e-4

        elif pred < self.sell_threshold:
            # SELL: if no shares, treat as HOLD (per request) — no penalty
            if self.shares <= 1e-12:
                # hold (no-op)
                pass
            else:
                frac = min(self.max_trade_fraction, max(0.0, -pred))
                qty = self.shares * frac
                if qty > 1e-12 and prev_shares > 0:
                    exec_price = price * (1.0 - self.slippage_pct)
                    proceeds = qty * exec_price * (1.0 - self.transaction_cost_pct)

                    # portion of inventory cost corresponding to sold qty
                    cost_reduced = (prev_inventory_cost) * (qty / prev_shares) if prev_shares > 0 else 0.0
                    # guard: cost_reduced must be <= prev_inventory_cost
                    cost_reduced = min(cost_reduced, prev_inventory_cost)

                    realized_profit = proceeds - cost_reduced

                    self.balance += proceeds
                    self.inventory_cost -= cost_reduced
                    # guard inventory_cost non-negative
                    if self.inventory_cost < 0:
                        self.inventory_cost = 0.0

                    self.shares -= qty

                    turnover_fraction = (qty * exec_price) / (prev_total + eps)
                    trade_count_this_step += 1
                else:
                    # treat as hold (no-op)
                    pass
        else:
            # HOLD
            pass

        # ---------- Portfolio & reward composition ----------
        current_total = self.balance + self.shares * price
        raw_delta = current_total - prev_total
        unrealized_change = raw_delta - realized_profit

        # normalize by prev_total (gives denser, adaptive reward signal)
        realized_pct = realized_profit / (prev_total + eps)
        unrealized_pct = unrealized_change / (prev_total + eps)

        reward = REALIZED_WEIGHT * realized_pct + UNREALIZED_WEIGHT * unrealized_pct
        reward -= TURNOVER_PENALTY_COEF * turnover_fraction
        reward -= MIN_TRADE_PENALTY * trade_count_this_step

        # churn penalty (small)
        if hasattr(self, "prev_action") and self.prev_action is not None:
            # penalize sign flips and large changes
            if float(np.sign(pred)) != float(np.sign(self.prev_action)):
                reward -= CHURN_PENALTY
            else:
                reward -= CHURN_PENALTY * abs(pred - self.prev_action)

        # drawdown penalty
        self.peak_balance = max(getattr(self, "peak_balance", self.initial_balance), current_total)
        drawdown = (self.peak_balance - current_total) / (self.peak_balance + eps)
        reward -= DRAWDOWN_PENALTY_COEF * drawdown

        # record history
        self.history.append({
            "step": int(self.current_step),
            "timestamp": self.timestamps[int(self.current_step)],
            "price": price,
            "balance": float(self.balance),
            "shares": float(self.shares),
            "portfolio_value": float(current_total),
            "reward": float(reward),
            "pred": float(pred),
            "realized_profit": float(realized_profit),
            "turnover_fraction": float(turnover_fraction),
            "inventory_cost": float(self.inventory_cost),
        })

        # update trackers
        self.prev_total = float(current_total)
        self.prev_action = float(pred)

        # ---------- Step progression & terminal handling ----------
        self.current_step += 1
        self.steps_left -= 1
        terminated = (self.steps_left <= 0) or (self.current_step >= len(self.prices) - 1)
        self.done = bool(terminated)

        # Force liquidation at episode end (if shares > 0)
        if terminated and self.shares > 0.0:
            liq_idx = min(self.current_step, len(self.prices) - 1)
            final_price = float(self.prices[liq_idx])
            exec_price = final_price * (1.0 - self.slippage_pct)
            proceeds = self.shares * exec_price * (1.0 - self.transaction_cost_pct)

            cost_reduced = self.inventory_cost
            cost_reduced = min(cost_reduced, self.inventory_cost)
            final_realized = proceeds - cost_reduced

            self.balance += proceeds
            self.trades.append(("sell-liquidation", liq_idx, exec_price, float(self.shares)))
            self.shares = 0.0
            self.inventory_cost = 0.0

            # include final realized reward (normalized)
            reward += REALIZED_WEIGHT * (final_realized / (self.prev_total + eps))

            current_total = self.balance
            self.prev_total = float(current_total)
            self.history.append({
                "step": int(liq_idx),
                "timestamp": self.timestamps[int(liq_idx)],
                "price": float(final_price),
                "balance": float(self.balance),
                "shares": 0.0,
                "portfolio_value": float(current_total),
                "reward": 0.0,
                "pred": 0.0,
                "realized_profit": float(final_realized),
                "turnover_fraction": 0.0,
                "inventory_cost": 0.0,
            })

        obs = self._get_obs() if not terminated else np.zeros(self.window_size * self.num_features, dtype=np.float32)

        return obs, float(reward), bool(terminated), False, {}




    def get_portfolio_history(self):
        if hasattr(self, "history") and len(self.history) > 0:
            import pandas as pd
            hist = pd.DataFrame(self.history)
            return hist["timestamp"].to_numpy(), hist["price"].to_numpy(), hist["portfolio_value"].to_numpy()
        else:
            # fallback behaviour (same as original)
            bal = float(self.initial_balance)
            shares = 0.0
            pv = []
            trades_by_step = {}
            for t in self.trades:
                trades_by_step.setdefault(t[1], []).append(t)
            for i in range(len(self.prices)):
                if i in trades_by_step:
                    for typ, step, p, qty in trades_by_step[i]:
                        if typ.startswith("buy"):
                            bal -= qty * p * (1.0 + self.transaction_cost_pct)
                            shares += float(qty)
                        else:
                            bal += qty * p * (1.0 - self.transaction_cost_pct)
                            shares -= float(qty)
                pv.append(bal + shares * float(self.prices[i]))
            return self.timestamps[:len(pv)], self.prices[:len(pv)], np.array(pv)
        
        
        
        
        
        
        






def make_train_env():
    def _init():
        env = TradingEnvContinuous(
            prices=train_prices,
            timestamps=df_train["timestamp"].to_numpy(),
            volumes=train_vols,
            window_size=WINDOW_SIZE,
            initial_balance=INITIAL_BALANCE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            slippage_pct=SLIPPAGE_PCT,
            episode_length=EPISODE_LENGTH,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
            max_trade_fraction=MAX_TRADE_FRACTION,
        )
        return Monitor(env)
    venv = DummyVecEnv([_init])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv



def train_ppo(total_timesteps=TOTAL_TIMESTEPS, model_path=MODEL_PATH, reload=False):
    """Train PPO on the training data and save the model."""
    vec_env = make_train_env()

    policy_kwargs = dict(net_arch=[256, 256])

    if reload and os.path.exists(model_path):
        print("[train] Loading existing model and continuing training:", model_path)
        model = PPO.load(model_path, env=vec_env)
        model.set_env(vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            
            policy_kwargs=policy_kwargs,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENTROPY_COEFFICIENT,
            learning_rate=LEARNING_RATE,
        )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print("[train] Model saved to", model_path)
    vec_env.close()
    return model


def evaluate_model(model, n_episodes=10, verbose=True):
    """
    Evaluate the trained model on the test set using the non-vector env.
    Returns summary dict and per-episode histories.
    """
    action_counts = {0: 0, 1: 0, 2: 0}  # bucketed actions (hold, buy, sell)
    ep_results = []

    for ep in range(n_episodes):
        env = TradingEnvContinuous(
            prices=test_prices,
            timestamps=df_test["timestamp"].to_numpy(),
            volumes=test_vols,
            window_size=WINDOW_SIZE,
            initial_balance=INITIAL_BALANCE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            slippage_pct=SLIPPAGE_PCT,
            episode_length=EPISODE_LENGTH,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
            max_trade_fraction=MAX_TRADE_FRACTION,
        )
        obs, _ = env.reset()
        done = False
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_steps += 1

            pred = float(np.asarray(action).ravel()[0])
            if pred > BUY_THRESHOLD:
                action_counts[1] += 1
            elif pred < SELL_THRESHOLD:
                action_counts[2] += 1
            else:
                action_counts[0] += 1

            if ep_steps > len(test_prices):
                break

        hist_ts, hist_prices, hist_pv = env.get_portfolio_history()
        final_equity = float(hist_pv[-1]) if len(hist_pv) > 0 else env.balance
        total_profit = final_equity - INITIAL_BALANCE
        closed_trades = sum(1 for t in env.trades if str(t[0]).lower().startswith("sell"))
        win_rate = 0.0
        if closed_trades > 0:
            # crude placeholder: wins counted as sells present (refine as needed)
            win_rate = len([t for t in env.trades if str(t[0]).lower().startswith("sell")]) / closed_trades

        ep_results.append({
            "final_equity": final_equity,
            "total_profit": total_profit,
            "num_trades": closed_trades,
            "win_rate": win_rate,
            "history_ts": hist_ts,
            "history_price": hist_prices,
            "history_pv": hist_pv,
            "trades": list(env.trades),
        })

        if verbose:
            print(f"[EP {ep+1}/{n_episodes}] final_equity={final_equity:.2f} profit={total_profit:.2f} trades={closed_trades} steps={ep_steps}")

    avg_final = np.mean([r["final_equity"] for r in ep_results])
    avg_profit = np.mean([r["total_profit"] for r in ep_results])
    avg_trades = np.mean([r["num_trades"] for r in ep_results])
    avg_win_rate = np.mean([r["win_rate"] for r in ep_results])

    summary = {
        "avg_final_equity": float(avg_final),
        "avg_profit": float(avg_profit),
        "avg_trades": float(avg_trades),
        "avg_win_rate": float(avg_win_rate),
        "action_counts": action_counts,
        "episodes": ep_results,
    }

    print("ACTION COUNTS:", action_counts)
    print("AVERAGE final_equity:", avg_final, "avg_profit:", avg_profit, "avg_trades:", avg_trades, "avg_win_rate:", avg_win_rate)
    return summary


# -------------------------
# Run training and evaluation
# -------------------------
if __name__ == "__main__":

    if os.path.exists(MODEL_PATH):
        print(f"[main] Found existing model at {MODEL_PATH}, loading and continuing training…")
        model = train_ppo(total_timesteps=TOTAL_TIMESTEPS,
                        model_path=MODEL_PATH,
                        reload=True)
    else:
        print("[main] No existing model found, training new PPO model…")
        model = train_ppo(total_timesteps=TOTAL_TIMESTEPS,
                        model_path=MODEL_PATH,
                        reload=False)


    summary = evaluate_model(model, n_episodes=10, verbose=True)

    with open(os.path.join(MODEL_DIR, "eval_summary.json"), "w") as f:
        json.dump({
            "summary": summary,
            "config": {
                "WINDOW_SIZE": WINDOW_SIZE,
                "EPISODE_LENGTH": EPISODE_LENGTH,
                "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
                "INITIAL_BALANCE": INITIAL_BALANCE,
                "TRANSACTION_COST_PCT": TRANSACTION_COST_PCT,
                "SLIPPAGE_PCT": SLIPPAGE_PCT,
            }
        }, f, default=lambda x: str(x))

    print("[main] Done. Results saved to", MODEL_DIR)
