import pandas as pd
import numpy as np
from scipy.stats import norm


class VpinStrategy:
    """Trading strategy implementation that produces trading signals time series.

    Strategy is based on two parameters - VPiN (The Volume Synchronized Probability of
    INformed Trading) and Quotation Imbalance as per Flow Toxicity and Volatility (http://ssrn.com/abstract=1695596),
    David Easley (Cornell University), Marcos López de Prado (Tudor Investment Corp., RCC at Harvard University) and
    Maureen O'Hara (Cornell University).

    Attributes:
        RESAMPLE_PERIOD (str): Timeframe at which the strategy operates (as per pandas definition).
        WINDOW_LENGTH (str): Timebars window length.
        BUCKET_SIZE (int): Size of the base asset's trading volume that is considered to be "standard".

    """

    RESAMPLE_PERIOD = "1min"
    WINDOW_LENGTH = 50
    BUCKET_SIZE = 1_000_000

    def __init__(
        self,
        vpin_threshold: float,
        imbalance_threshold: float,
        weight_increase: float,
        max_leverage: float,
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
    ):
        """__init__ method.

        Args:
            vpin_threshold (float): Threshold for VPIN-based "Buy"/"Sell" signal, hyperparameter.
            imbalance_threshold (float): Threshold for QuoteImbalance-based "Buy"/"Sell" signal, hyperparameter.
            weight_increase (float): Level by which the position changes (in %), when there is trading
                                     signal, hyperparamemeter.
            max_leverage (float): Maximum level of leverage, allowed in the strategy.
            trades (pd.DataFrame): Dataset of trades data that should include VWAP prices and volumes for each
                                   timestamp.
            quotes (pd.DataFrame): Dataset of orderbook VWAP bid-ask quotes (already calculated)

        """
        self.vpin_threshold = vpin_threshold
        self.imbalance_treshold = imbalance_threshold

        self.weight_increase = weight_increase
        self.max_leverage = max_leverage

        self.trades = trades
        self.quotes = quotes

        # Initialize attributes that hold data in the instance
        self._dataset = None
        self._weights = None

    def _get_z_score(self, feature: pd.Series, max_sigmas: float = 3.0) -> pd.Series:
        """Calculate z-score for given distribution of values and restrict by max_sigmas value.

        Args:
            feature (pd.Series): Some feature value, assumed to follow chosen distribution.
            max_sigmas (float): Maximum number of sigmas that is considered to be critical for chosen distribution.

        Returns:
            pd.Series.

        """
        # Get rolling mean and std
        mean = feature.rolling(window=self.WINDOW_LENGTH).mean()
        std = feature.rolling(window=self.WINDOW_LENGTH).std()

        # Compute z-scores
        z_scores = (feature - mean) / std
        z_scores[z_scores > max_sigmas] = max_sigmas
        z_scores[z_scores < -max_sigmas] = -max_sigmas

        return z_scores

    def _init_dataset(self) -> None:
        """Initialize the dataset from trades and quotes data (in-place)."""
        # Dispense from the timestamp UTC
        self.trades["time"] = pd.to_datetime(self.trades["time"]).dt.tz_localize(None)
        self.quotes["time"] = pd.to_datetime(self.quotes["time"]).dt.tz_localize(None)

        # Order trades and quotes
        self.trades.sort_values("time", inplace=True)
        self.quotes.sort_values("time", inplace=True)

        # Reset indices and merge two datasets by nearest timestamp value
        # inside (!) a minute (will trade next minute, so no bias produced)
        quotes_reindex = (
            self.quotes.set_index("time")
            .reindex(self.trades.set_index("time").index, method="nearest")
            .reset_index()
        )
        self._dataset = pd.merge(self.trades, quotes_reindex, on="time").set_index(
            "time"
        )

    def _add_quote_volumes(self) -> None:
        """Add bid and ask quotation volumes from share_of_bids data (in-place)."""
        self._dataset["bid_volume"] = (
            self._dataset["volume_traded"] * self._dataset["share_of_bids"]
        )
        self._dataset["ask_volume"] = self._dataset["volume_traded"] * (
            1 - self._dataset["share_of_bids"]
        )

    def _create_buckets(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Create volume buckets in order to produce discrete quotation data.

        Args:
            df_in (pd.DataFrame): Raw quotes dataset.

        Returns:
            pd.DataFrame.

        """
        # Initialize counter
        count = 0
        buy_volume = 0
        sell_volume = 0
        buckets_series = []
        for index, row in df_in.itertuples():
            new_volume = row.volume
            z = row.z

            # Fill the bucket until volume is reached
            if self.BUCKET_SIZE < count + new_volume:
                # Get supposed buy and sell volumes by z-value (assuming that large deviations are caused by toxic flow)
                buy_volume = buy_volume + (self.BUCKET_SIZE - count) * z
                sell_volume = sell_volume + (self.BUCKET_SIZE - count) * (1 - z)

                buckets_series.append(
                    {"buy": buy_volume, "sell": sell_volume, "time": index}
                )
                count = new_volume - (self.BUCKET_SIZE - count)

                # If bucket is not full, continue filling
                if int(count / self.BUCKET_SIZE) > 0:
                    for i in range(0, int(count / self.BUCKET_SIZE)):
                        buy_volume = self.BUCKET_SIZE * z
                        sell_volume = self.BUCKET_SIZE * (1 - z)
                        buckets_series.append(
                            {"buy": buy_volume, "sell": sell_volume, "time": index}
                        )

                count = count % self.BUCKET_SIZE
                buy_volume = count * z
                sell_volume = count * (1 - z)
            else:
                buy_volume = buy_volume + new_volume * z
                sell_volume = sell_volume + new_volume * (1 - z)
                count = count + new_volume

        return pd.DataFrame(buckets_series).set_index("time")

    def _calculate_vpin(self, resample_period: str) -> None:
        """Calculate VPiN metric and produce CDF for its values (in-place). Shows order flow toxicity.

        Args:
            resample_period (str): Timeframe at which the strategy operates (as per pandas definition).

        Raises:
            AssertionError: If the dataset provided is not a pandas.DataFrame.

        """
        if self._dataset is None:
            self._init_dataset()

        assert isinstance(
            self._dataset, pd.DataFrame
        ), f"Expected {pd.DataFrame}, got {type(self._dataset)}"

        # Get resampled data
        trades_resampled = (
            self._dataset["vwap_traded"]
            .diff(1)
            .resample(resample_period)
            .sum()
            .dropna()
        )
        volume_resampled = (
            self._dataset["volume_traded"].resample(resample_period).sum().dropna()
        )
        sigma = trades_resampled.std()
        z = trades_resampled.apply(lambda x: norm.cdf(x / sigma))

        # Create z-values
        vpin_df = pd.DataFrame({"z": z, "volume": volume_resampled}).dropna()

        # Collect buckets
        buckets = self._create_buckets(vpin_df)

        # Calculate VPiN by rolling window bucket supposed buy-sell deviations
        buckets["VPIN"] = (
            abs(buckets["buy"] - buckets["sell"]).rolling(self.WINDOW_LENGTH).mean()
            / self.BUCKET_SIZE
        )
        buckets["VPIN_CDF"] = buckets["VPIN"].rank(pct=True)

        buckets = (
            buckets.reset_index()
            .drop_duplicates(subset="time", keep="last")
            .set_index("time")
        )
        buckets = buckets.dropna(subset=["VPIN_CDF"])

        self._dataset = self._dataset.join(buckets)

    def _calculate_quote_imbalance(self) -> None:
        """Calculate quotation imbalance metric (difference between bid-ask volumes times bid-ask prices)
        and produce z-scores for its values (in-place). Shows skewness of the market quotes (in-place).

        """
        avg_bid = self._dataset["bid_volume"] * self._dataset["bid"].apply(
            lambda x: int(x != 0)
        )
        avg_ask = self._dataset["ask_volume"] * self._dataset["ask"].apply(
            lambda x: int(x != 0)
        )

        imbalance = avg_bid - avg_ask

        self._dataset["imbalance"] = self._get_z_score(imbalance)

    def _produce_weights(self):
        """Calculate weights for the asset in the strategy, based on features produced (in-place)
        and strategy hyperparameters.

        Note: Strategy does not require training on historical data, so the weights are set by heuristic rules
              (as per paper).

        """
        weights = []

        position = 0
        buy_count = 0
        sell_count = 0
        for index, row in self._dataset.itertuples():
            # Trade only if the flow is toxic enough (informed trading signal), i.e. VPiN is high enough
            if row.VPIN_CDF > self.vpin_threshold:
                # If imbalance is skewed up, need to buy
                if row.imbalance > self.imbalance_treshold:
                    if position < 0:
                        position = 0

                    # Add gradually weights according to the set hyperparameter
                    position += self.weight_increase
                    buy_count = 10

                # If imbalance is skewed down, need to sell
                if row.imbalance < -self.imbalance_treshold:
                    if position > 0:
                        position = 0

                    # Add gradually weights according to the set hyperparameter
                    position -= self.weight_increase
                    sell_count = 10

            if position > 0:
                buy_count = buy_count - 1
                if buy_count == 0:
                    position = 0
            elif position < 0:
                sell_count = sell_count - 1
                if sell_count == 0:
                    position = 0

            weights.append({"position": position, "time": index})

        weights = self._restrict_weights_by_leverage(
            pd.DataFrame(weights).set_index("time")
        )
        self._weights = self._dataset.join(weights)

    def _restrict_weights_by_leverage(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Apply restriction of maximum leverage by setting lower and upper bounds on weights.

        Args:
            weights (pd.DataFrame): Time series of weights, produced by the strategy logic.

        Returns:
            pd.DataFrame

        """
        weights = np.where(
            weights.to_numpy() > self.max_leverage, self.max_leverage, weights
        )
        weights = np.where(weights < -self.max_leverage, -self.max_leverage, weights)
        return pd.DataFrame(weights)

    def get_trading_strategy(self) -> pd.DataFrame:
        """Produce weights for the asset, according to the VpinStrategy trading rules.

        Returns:
            pd.DataFrame

        """
        self._init_dataset()
        self._add_quote_volumes()

        self._calculate_vpin(resample_period=self.RESAMPLE_PERIOD)
        self._calculate_quote_imbalance()
        self._produce_weights()

        return self._weights
