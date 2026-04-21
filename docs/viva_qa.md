# Common Viva Questions and Answers

**Q1: Why did you choose LSTM over standard RNN?**
**A1:** Standard RNNs suffer from the "vanishing gradient problem," making them unable to learn long-term dependencies. Financial data relies heavily on long-term trends. LSTMs solve this by introducing a "cell state" and three gates (Input, Output, Forget) that allow the network to retain important information over long sequences and forget irrelevant noise.

**Q2: What is the Vanishing Gradient Problem?**
**A2:** During backpropagation in deep networks, gradients are multiplied. If gradients are small (< 1), they exponentially shrink as they propagate backward through the layers, eventually becoming zero ("vanishing"). As a result, early layers fail to learn. LSTMs mitigate this using their additive cell state.

**Q3: How did you ensure there was no "Data Leakage"?**
**A3:** Data leakage in time-series forecasting occurs if future information leaks into the training dataset. 
1. We did not use random `train_test_split`. We strictly split data chronologically (e.g., first 80% of time for training, last 20% for testing).
2. We fitted the `MinMaxScaler` **only** on the training data. Then we used that exact same scaler to transform both train and test sets.

**Q4: Why use a MinMaxScaler and not StandardScaler?**
**A4:** Neural networks (especially those with Sigmoid or Tanh activations in their gates, like LSTM) perform best when inputs are neatly bounded between 0 and 1 or -1 and 1. Stock prices can vary wildly (from $10 to $3000), and MinMax brings all features to a uniform 0-1 scale, ensuring the model converges faster and no single feature dominates the learning process.

**Q5: What is the purpose of the Dropout layer?**
**A5:** Dropout is a regularization technique. It randomly "turns off" a percentage of neurons during each training epoch so the network does not become overly reliant on any specific path. It forces the network to learn generalized patterns, significantly reducing overfitting—which is critical because stock data is extremely noisy.

**Q6: What is a Lookback Window (or time step)? Why 60 days?**
**A6:** It represents how many previous days of data the model uses to predict the current day. 60 days roughly equates to 3 "trading months." It provides enough medium-term historical context for the model to establish a trend without being bogged down by years of potentially irrelevant old data.

**Q7: Explain the MACD and RSI indicators.**
**A7:** 
- **RSI (Relative Strength Index):** Measures momentum on a scale of 0 to 100. Over 70 suggests the stock is "overbought" (might fall soon) and under 30 suggests "oversold" (might rise soon).
- **MACD (Moving Average Convergence Divergence):** Shows the relationship between two moving averages (usually 12-day and 26-day EMA). When the MACD line crosses its signal line, it suggests a changing trend.

**Q8: Why did you use VADER for Sentiment Analysis?**
**A8:** VADER is specifically designed for short, informal text and heavily relies on a predefined lexicon. It understands punctuation intensity (e.g., "!!!" makes a score stronger) and handles capitalization well. This makes it highly effective for parsing short financial news headlines compared to general-purpose NLP tools.

**Q9: What is the Loss Function and Optimizer used?**
**A9:**
- **Optimizer: Adam.** It adapts the learning rate during training, offering faster convergence than basic Stochastic Gradient Descent.
- **Loss Function:** Mean Squared Error (MSE). We use MSE because this is a regression problem (predicting a continuous value, price), and MSE heavily penalizes large outlier errors.

**Q10: Why is stock prediction generally considered impossible to perfect?**
**A10:** The Efficient Market Hypothesis suggests that asset prices fully reflect all available information. Market movements are largely driven by unpredictable real-world events (politics, natural disasters, sudden earnings reports, CEO tweets) that are not present in historical price data. Our model captures *trends* and *momentum*, not unpredictable macroeconomic shocks.
