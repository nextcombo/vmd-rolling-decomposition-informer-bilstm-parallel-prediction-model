# VMD Rolling Decomposition + Informer-BiLSTM Parallel Prediction Model

\tPreface  
\tIn the task of time series prediction, the use of decomposition algorithms like EMD (Empirical Mode Decomposition), CEEMDAN (Complete Ensemble Empirical Mode Decomposition), and VMD (Variational Mode Decomposition) may introduce information leakage. The specific situation depends on the application methods of these methods. The main risk of information leakage is that future information is leaked to the model, making the model perform better in practical applications than it should.  
\tTo prevent information leakage, we first divide the dataset before decomposition. Then, we use a sliding window method to create dataset labels, and finally gradually roll the time series window for decomposition!  
1. Innovation Model Effect:  
\t1.1 Introduction to Relevant Experiments:  
Based on a wind power power data set, we propose a parallel prediction model of VMD+Informer-BiLSTM, which significantly improves the prediction effect in single step prediction tasks!  
\tUnder the condition of setting the sliding window value at 96 steps, we perform VMD decomposition on each sample window value and provide comparative experiments under different component conditions:  
The results are as follows:  
With the increase of IMF component number, more frequency information can be captured, and each IMF corresponds to different time scales, providing richer features for the prediction model. The model fitting score R2 shows an upward trend, while there is also a smaller MSE.  
\t1.2 Model Evaluation:  
\t  
\t1.3 Wind Power Prediction Visualization:  
\t  
We also provide experiments and comparisons based on multiple datasets in multiple prediction tasks, including detailed materials, explanation documents, and video tutorials. This includes how to replace your own dataset, parameter adjustment tutorials, and how to replace prediction tasks, with code commented line by line and detailed parameter introductions:  
● Data Sets: Wind farm wind power data set, power data set, wind speed data set, etc.  
● Environment Framework: Python 3.9, PyTorch 1.8 and above versions can run.
2.1 Model Innovation Points
- Combine the advantages of Informer and RNN
- Informer: Proficient in handling long sequences, capable of parallel computation, enhancing computational efficiency and prediction performance. It is improved on the basis of Transformer to better suit time series data, especially those with long dependencies.
- BiLSTM: Outperforms in capturing both short-term and long-term dependencies in sequence data, effectively managing temporal relationships within the sequence data.
- By parallelly using these two models, patterns at different temporal scales can be better captured, improving prediction accuracy and robustness.
- 2.2 Parallel Model Architecture
- Parallel use of Informer and BiLSTM allows for information extraction and processing at different temporal scales through two branches working in parallel:
  - Informer part: Processes global time series patterns, effectively handling long sequence data.
  - BiLSTM part: Processes local time series patterns, effectively capturing short-term dependencies and dynamic changes in sequence data.
- This architecture comprehensively captures the characteristics of time series data, enhancing model prediction performance.
- 2.3 Model Fusion
- Concatenate the outputs of Informer and BiLSTM into a single fully connected layer to fuse features from different models. This fusion approach enables the model to simultaneously utilize the global information extraction capability of Informer and the local time series relationship modeling capability of LSTM.
2.4 Efficient Computation
The use of Informer significantly enhances the computational efficiency for long-term sequential data, while the use of BiLSTM ensures the full utilization of local temporal information. This combination not only guarantees efficient computation but also improves prediction accuracy and reliability.
3 Informer in-depth explanation with three innovative points
3.1 Probability - Sparse Attention Mechanism (ProbSparse Self-attention)
ProbSparse Self-attention is a sparse attention mechanism introduced in the Informer model. The core idea is to select the most important part of attention weights through probability methods, ignoring those that have a smaller impact on the results. This method can significantly reduce computational complexity while maintaining high model performance.
Sparse Attention: Unlike the dense self-attention mechanism of standard Transformers, Informer introduces ProbSparse Self-attention, which selects important Q-K pairs through a probability sampling mechanism, reducing computational complexity.
Efficiency Improvement: The sparse attention mechanism significantly reduces computational complexity from O(L2⋅d) to O(L⋅log(L)⋅d), where L is the sequence length and d is the feature dimension per time step.
3.2 Multi-scale Feature Extraction - Information Distillation
Unlike Transformers, which mark N encoder stacks on the left side of the Encoder architecture for each encoder, Informer's architecture diagram shows two trapezoids, one large and one small, representing the main stack of the entire input sequence.
The Encoder's role is Self-attention Distilling. Due to the ProbSparse self-correlation mechanism often using V's mean for filling, there naturally exists redundant attention source. Therefore, convolution and pooling are applied between adjacent Attention Blocks to downsample features, which is why the author designs the Encoder by continuously extracting key features through distillation operations, thus obtaining feature maps worth focusing on.

Multi-scale time series feature extraction: Informer models different time granularities of features through multi-scale methods, better capturing the multi-scale dependencies in time series.

Information distillation: The introduction of information distillation mechanism, through hierarchical time convolutional pooling layers, progressively reduces the time step size to extract features at different scales, achieving efficient modeling of long-term dependencies.

Convolutional dimensionality reduction: A 1D convolutional pooling layer is used in the encoder for dimensionality reduction, with a stride of 2, reducing the sequence length by half, further reducing computational complexity.

Information compression: Through convolutional pooling layers, information is compressed, condensing long sequence information into shorter time steps, thereby more efficiently performing temporal modeling.

3.3 Time encoding
Informer not only adds PositionEmbedding (position encoding) essential for the Transformer architecture but also adds various encodings related to time:

Day cycle encoding: Represents the time points in a day.

Week cycle encoding: Represents the time points in a week.

Month cycle encoding: Represents the time points in a month.

In the LSTF problem, the ability to capture remote independence requires global information, such as hierarchical timestamps (week, month, and year) and unknowable timestamps (holidays, events).
1. Specific requirements for the GlobalTimeStamp need to be confirmed based on actual problems. For example, in calculating passenger flow at high-speed train stations, the "holiday" time difference is obviously very important. If calculating the passenger flow of commuter transportation such as buses and subways, it's clear that "weekday" can reveal whether it's a working day.
4. Data Set Introductions
4.1 Importing Data
The wind power dataset consists of 35040 samples with 15 features, and we select the first 6000 data points for visualization.
4.2 Other Data Set Introductions
Refer to previous articles:
Freely available | Time Series Common Data Sets, Visualization Codes
4.3 Data Set Production and Preprocessing
Detailed explanations are provided in the documents provided!
5. Parallel Prediction Model Based on VMD+Informer-BiLSTM
5.1 Definition of Informer-BiLSTM parallel prediction network model
5.2 Setting parameters and training the model
Trained for 50 epochs, MSE is 0.01283, the VMD+Informer-BiLSTM parallel prediction effect is significant. The model can fully utilize the long-term dependency modeling capability of Informer and the short-term dependency capturing capability of BiLSTM, converges quickly, has excellent performance, high prediction accuracy, and further improves the model prediction performance by appropriately adjusting model parameters.
6. Results Visualization and Model Evaluation
6.1 Prediction results visualization
6.2 Model evaluation
The predicted results show that under the VMD+Informer-BiLSTM parallel prediction model, the fitting effect is good. Through this design, we can fully utilize the advantages of Informer and BiLSTM to achieve efficient and accurate time series prediction, achieving significant combined prediction effects!
Previously interesting content:
Time series forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed prediction (I) Data introduction and preprocessing
Wind speed prediction (II) EMD-LSTM model based on Pytorch
Wind speed prediction (III) EMD-LSTM-Attention model
Wind speed prediction (IV) EMD-Transformer model based on Pytorch
Wind speed prediction (V) EMD-CNN-LSTM model based on Pytorch
Wind speed prediction (VI) EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + combined forecasting model (BiLSTM-Attention + ARIMA)
CEEMDAN + combined forecasting model (CNN-LSTM + ARIMA)
CEEMDAN + combined forecasting model (Transformer - BiLSTM + ARIMA)
CEEMDAN + combined forecasting model (CNN-Transformer + ARIMA)
Multi-feature variable sequence prediction (I) - CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (II) - CNN-LSTM-Attention wind speed prediction model
Multi-feature variable sequence prediction (III) - CNN-Transformer wind speed prediction model
Multi-feature variable sequence prediction (IV) Transformer-BiLSTM wind speed prediction model
Multi-feature variable sequence prediction (V) CEEMDAN+CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (VI) CEEMDAN+CNN-Transformer wind speed prediction model
Prediction model based on SSA optimized by Sparrow optimization algorithm for CEEMDAN-BiLSTM-Attention
CEEMDAN-Transformer-BiGRU Prediction Model Based on Optimized Algorithm SSA for Sparrows
Super Predictive Model: Secondary Decomposition-Combination Prediction
Multi-feature variable sequence prediction (7) CEEMDAN+Transformer-BiLSTM prediction model
Multi-feature variable sequence prediction (8) CEEMDAN-SSA-BiLSTM prediction model based on the Sparrow Optimization Algorithm
Multi-feature variable sequence prediction (11) TCN-GRU prediction model based on Pytorch
VMD + CEEMDAN secondary decomposition, CNN-LSTM prediction model
VMD + CEEMDAN secondary decomposition, CNN-Transformer prediction model
VMD + CEEMDAN 2nd Decomposition, Transformer-BiGRU Prediction Model
Predictive Model Based on the Optimized Algorithm of Sparrows - Code All-in-One
Super-strong combination prediction model!
High Innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention Predictive Model
Super-strong combination prediction model!
Lowest price on the net | Family Packs are continuously updated!
| Based on TCN-SENet + BiGRU-GlobalAttention Parallel Prediction Model
Secondary Decomposition - Innovative Model Prediction for the Family Pack
BiTCN-BiGRU-CrossAttention fusion of spatiotemporal features high innovative prediction model
Combinatorial Prediction Update | Superior Combinatorial Predictive Model!
BiTCN-Transformer model for parallel prediction with fusion of spatiotemporal features
Parallel prediction model based on multi-level attention mechanism
CEEMDAN-CNN-GRU-GlobalAttention + XGBoost ensemble prediction
Multi-step prediction series | LSTM, CNN, Transformer, TCN, serial, parallel model collection
END
\t2. Friends who like it can click to follow, give the article a "like", share on social media or discuss in groups.

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
