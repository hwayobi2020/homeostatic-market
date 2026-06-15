# -*- coding: utf-8 -*-
"""v0.8 한글 docx -> 영어 번역본(_End.docx). 텍스트 run의 글자만 교체.
스타일/폰트/그림(drawing)/표 구조는 일절 건드리지 않음 (w:t 텍스트만 수정).
인덱스 기반 (iter_paras 순서 = 추출 때와 동일)."""
import docx
from docx.oxml.ns import qn

SRC = 'Monetary Debasement and Equity Tail-Risk A Short-Rate Path Conditional Normalizing Flow Approach_v0.8.docx'
DST = 'Monetary Debasement and Equity Tail-Risk A Short-Rate Path Conditional Normalizing Flow Approach_v0.8_End.docx'


def iter_paras(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p


def set_para_text(p, en):
    """text run 의 w:t 글자만 교체. drawing 포함 run·폰트·스타일 보존."""
    text_runs = [r for r in p.runs if r._r.findall(qn('w:t'))]
    if not text_runs:
        return
    text_runs[0].text = en
    for r in text_runs[1:]:
        r.text = ""


TRANS = {
 3: "Through the 2008 financial crisis and the COVID-19 pandemic, unconventional monetary policy became a standing tool and triggered excess liquidity, raising debates over the debasement of money's purchasing power and asset inflation. However, because of the nature of capital markets, observable Tail-Risk data corresponding to monetary policy and liquidity expansion are limited. For this reason, counterfactual scenarios become a meaningful analytical tool. Yet existing deep generative models in finance have focused on prediction, so cases that consider counterfactual conditions are limited, and those that further consider conditional paths are even rarer. This study proposes MAC-Flow, an autoregressive (AR) Normalizing Flow model that generates counterfactual equity scenarios for the next 13 weeks from the conditional time-series trajectories of the 3M T-bill rate, which is closely correlated with the U.S. Federal Reserve (Fed) policy rate, and an excess-liquidity indicator (M2-INDPRO-CPI). Built from a Past encoder that compresses the historical series and a Per-step encoder, the model reproduces scenarios in which tail risk widens under the combination of low rates and liquidity expansion during crises, and confirms that excess liquidity and the short-term rate are meaningful variables for tail-risk measurement. It also shows that, even under identical mean and variance assumptions, the intra-horizon loss, an investor's effective risk, differs according to the shape of the rate and liquidity paths. As an equity-scenario analysis tool linked to the path of monetary policy, which strongly affects financial markets, this model can be usefully applied to portfolio risk management and monetary-policy analysis.",
 5: "Through the 2008 global financial crisis and the 2020 COVID-19 pandemic, unconventional monetary policy represented by quantitative easing and zero interest rates moved beyond one-off crisis responses to become an established policy instrument, and the U.S. Federal Reserve's (Fed) balance sheet underwent years of cumulative expansion. This normalization of unconventional monetary policy gave rise to debates over the debasement of money's purchasing power (monetary debasement). Friedman, in classical monetary theory, held that changes in the money supply are ultimately reflected in inflation and asset prices, and Minsky's financial instability hypothesis warned that leverage and risk-taking accumulated during stable phases ultimately raise systemic fragility. In this way, liquidity injection defends the mean of asset returns in the short run while simultaneously carrying the dilemma of policy-induced fragility that enlarges left-tail fragility.",
 6: "As the Lucas critique implies, macro-financial dynamics are difficult to analyze from historical data alone. Capital markets, by their nature, make it hard to secure sufficient data, and the same policy can produce different market reactions depending on the regime. For these reasons, counterfactual scenario generation becomes a meaningful analytical tool, because by assuming policy paths or liquidity states that were not actually observed, one can generate the equity distribution that could arise under those conditions as scenarios, thereby extending the reach of risk management.",
 7: "Counterfactual scenarios in finance require a time-series model capable of producing flexible distributions. Traditionally, GARCH-family volatility models have consistently captured volatility clustering through autoregressive variance dynamics, but they are limited in expressing flexible distributions. More recently, deep generative models such as VAE and GAN in finance can produce flexible distributions, but most are sampling or forecasting models, and even those that reflect counterfactual conditions did not consider path dependence. To overcome these limitations, this study proposes the Macro-Path-Conditional Counterfactual Flow (MAC-Flow) model. The main contributions of this study are as follows.",
 8: "First, we propose MAC-Flow, a path-dependent counterfactual scenario generator that simultaneously reflects monetary policy and the debasement of money. MAC-Flow is an AR-based deep generative model capable of flexible conditional distributions and density estimation, and it measures intra-horizon loss, a path-dependent risk that prior models in finance overlooked.",
 9: "Second, we show that the macro-equity co-movement, in which left-tail risk is amplified under the combination of low rates and a liquidity surge, is reproduced in the model's scenarios, and we confirm that, by using the time-series paths of the excess-liquidity indicator and the short-term rate as counterfactual conditions, meaningful counterfactual scenarios can be generated across various market regimes, including crises in which tail risk widens.",
 12: "It is well known that the policy rate has a major effect on equity movements. Bernanke and Kuttner (2005) split FOMC decisions into expected and unexpected changes and quantified that unexpected policy-rate shocks move equity prices significantly. Adrian and Shin (2010) quantified the channel through which risk-taking triggered by low rates is amplified into asset-price tail risk via the coupling of liquidity and leverage.",
 13: "The effect of the money supply on asset prices is rooted in a long tradition of monetary economics. Pearce and Roley (1985) reported that changes in the money supply shock equity prices, Adalid and Detken (2007) showed that excess money growth above average that is not absorbed by real output or prices, that is, excess liquidity, is systematically associated with asset-price boom-bust phases, and Belke et al. (2010) presented a channel through which the expansion of global liquidity lifts asset prices.",
 14: "The impact of monetary policy and liquidity depends not only on the level at a given point but on the time-series path. Woodford (2012), through research on forward guidance, established that the trajectory of rate policy rather than its level dominates the formation of financial conditions, and Cieslak (2019) showed that the risk premium in capital markets is adjusted continuously along the flow of inter-meeting expectations between FOMC meetings. Gurkaynak, Sack, and Swanson (2005) decomposed FOMC statements into a current rate shock (target factor) and expectations about the future policy path (path factor), and demonstrated empirically that asset prices respond far more to expectations about the future policy path than to the immediate rate decision. On the liquidity side as well, D'Amico and King (2013) showed that the effect of large-scale asset purchases depends not only on the stock of holdings but on the flow of supply while purchases are ongoing, that is, the path of liquidity injection, and the financial-cycle literature represented by Borio (2014) showed that the cumulative evolution of credit and liquidity is systematically associated with asset-price boom-bust.",
 16: "The academic tradition of analyzing the effect of macro shocks on asset prices through scenarios was established with the Vector Autoregression (VAR) and Impulse Response Function (IRF) of Sims (1980, Econometrica), and the Structural VAR of Bernanke (1986) and others systematized counterfactual intervention against exogenous shocks. However, VAR models remain within parametric and linear structures and are limited in expressing the nonlinearity of equity prices.",
 17: "In the tradition of modeling equity-market volatility, the GARCH family holds an important position. Starting from the ARCH of Engle (1982, Econometrica), it was generalized to the GARCH of Bollerslev (1986, J. Econometrics), and was later extended toward absorbing asymmetry and leverage effects and integrating exogenous variables. As a case of directly coupling macro variables into volatility dynamics, GARCH-MIDAS (Engle, Ghysels, and Sohn 2013, Rev. Econ. Stat.) integrated monthly macro variables into the long-run component of volatility through mixed-data sampling. However, the GARCH family integrates the effect of variables only at the level of the volatility scale, so the shape of the distribution is hard to change according to conditions.",
 19: "Deep generative models for time series have developed from the variational autoencoder (VAE) family. A VAE is a generative model composed of an encoder that compresses data into a probability distribution in a low-dimensional latent space and a decoder that reconstructs and generates data from those latent variables, and it can create new sequences by drawing samples from the latent space. TimeVAE advanced latent representation learning and generation of time-series sequences, and beta-VAE advanced disentanglement of latent variables. MacroVAE (ICAIF 2025) showed a case of generating macro-conditional equity-return sequences by injecting macro variables such as inflation and growth into the conditioning layer of the decoder.",
 20: "Generative adversarial network (GAN) based models also form one axis of generative models. A GAN is a model that trains a generator, which creates samples resembling the real ones, and a discriminator, which tries to distinguish real from fake, in competition (adversarially), so that the generator gradually produces data hard to distinguish from the real. TimeGAN (Yoon et al. 2019, NeurIPS) proposed synthetic data generation specialized in preserving the temporal structure of time series, and Quant GANs (Wiese et al. 2020, Quantitative Finance) specialized in reproducing the heavy tails, autocorrelation, and volatility clustering of financial returns.",
 21: "Diffusion models have also been introduced into the time-series generative-model domain. A diffusion model learns, in reverse, the process of gradually adding noise to data until it becomes pure noise, and generates new data by starting from noise and removing it step by step (denoising). MARCD (2025) produced regime-specific asset-distribution scenarios through a diffusion generator conditioned on latent regimes, and the Diffusion Factor Model (2025) integrated a latent factor structure into the diffusion process, enabling factor-dependent simulation of high-dimensional assets.",
 22: "The VAE, GAN, and diffusion models applied as deep generative models have the advantage of expressing flexible distributions, but most were not used for counterfactual scenario generation, and even cases that take macro variables as counterfactual conditions (MacroVAE) injected them as fixed scalar or vector conditions rather than as time-varying paths.",
 24: "Path-dependent counterfactual-estimation deep learning models have developed in the field of causal inference. The Counterfactual Recurrent Network (Bica et al. 2020) and the Causal Transformer (Melnychuk et al. 2022), among others, take time-varying treatment trajectories in medicine as step-wise inputs and estimate outcome trajectories while adjusting for time-varying confounding. These models mainly aim at point estimation of the expected state of an individual unit.",
 25: "Step-wise distribution learning with an autoregressive (AR) structure has developed in the forecasting field. Rasul et al. (2021, ICLR) compressed past time-series information with an RNN and, using a Normalizing Flow in an AR rollout conditioned on the hidden state, forecast multivariate probability distributions in domains such as electricity and traffic, and DeepAR (Salinas et al. 2020) and the Temporal Fusion Transformer (Lim et al. 2021) presented multi-horizon distributional forecasting models that take pre-available information (exogenous covariates) as future-trajectory inputs.",
 26: "The MAC-Flow of this study combines the AR-rollout approach of deep generative forecasting models with the counterfactual perspective of causal inference. Beyond mere machine-learning performance, this structure quantifies how equity-market risk unfolds along the trajectory of the macroeconomy and monetary policy under excess-liquidity shocks, and provides a tool for assessing equity tail risk under counterfactual policy and liquidity scenarios.",
 28: "Traditionally, since Markowitz's (1952) mean-variance framework, the standard deviation has been used as the basic measure of equity risk, but by treating upside and downside symmetrically it fails to distinguish the left tail (downside). To complement this, Value-at-Risk (VaR; RiskMetrics), which looks at the loss quantile at a given confidence level, and its conditional expectation CVaR / Expected Shortfall (Rockafellar and Uryasev 2000; Acerbi and Tasche 2002) have been widely used.",
 29: "However, Kritzman and Rich (2002) pointed out that endpoint-based risk underestimates the risk actually experienced during the holding period, and proposed the concepts of within-horizon risk and continuous VaR, and Bakshi and Panayotov (2010) proposed intra-horizon risk based on first-passage probabilities.",
 30: "This study uses the intra-horizon loss (the worst loss within the holding period), that is, the lowest cumulative return relative to the entry point during the holding period, as the core tail-risk measure. This captures the path and order dependence whereby an investor's experience of intermediate losses differs even when the endpoint return is the same.",
 32: "3. Methodology (Proposed Architecture)",
 33: "3.1 Research Data and Preprocessing",
 34: "3.1.1 Data Coverage",
 35: "This study targets weekly data from January 1971 to 2025, a choice that reflects data availability and frequency. As shown in Table 3.1, the input features include the policy rate (3M T-bill rate) and the base data used to compute excess liquidity, namely the money supply (M2), industrial production (INDPRO), and consumer prices (CPI), as well as the auxiliary variables ADS Business Conditions Index and the WTI price.",
 36: "Table 3.1: Data Variables and Sources",
 37: "To block look-ahead leak, the actual publication lag was applied to macro statistics that have publication delays. A lag of 1 week was applied to ADS, and 2 weeks to CPI, INDPRO, and M2 (except that M2 uses 1 week after the change of its publication cycle in February 2021), while no lag was applied to market variables that are observable in real time (S&P500, 3M T-bill). In addition, forward fill was applied to the monthly-published INDPRO and CPI and to WTI, for which a monthly series was used for long-term coverage. For M2, however, a weekly series is not provided before January 1981, so linear interpolation was applied to that period.",
 38: "3.1.2 Time-Series Cross-Validation (walk-forward fold)",
 39: "As shown in Table 3.2, the model was trained and tested over a total of four periods in a walk-forward manner. A 6-month gap was placed between periods to block data leakage. The test data of each fold, constructed with an expanding window, correspond to the 2008 global financial crisis, the easing period, the COVID shock period, and the post-COVID period, and were configured to include macro regimes that are mutually distinct in terms of monetary policy, liquidity, and crisis. The model's input window includes 52 weeks of past returns, so the first 52 weeks of each period are consumed as input, after which it is applied in a rolling manner in units of 65 weeks (52 past weeks + 13 future weeks).",
 40: "Table 3.2: Walk-Forward Fold Definition",
 41: "3.2 Variable Definitions",
 42: "3.2.1 Target Variable",
 43: "The target series is the sequence of S&P500 weekly returns for the 13 weeks following the origin. The 13-week forecast horizon corresponds to about one quarter (3 months), spanning roughly two FOMC meetings, and is consistent with the time scale of the future policy path (Gurkaynak, Sack, and Swanson 2005).",
 44: "3.2.2 Conditional Variables (Conditional Path)",
 45: "As shown in Table 3.3, the model input is injected as the observed values of five channels over the past 52 weeks and the hypothetical (counterfactual) paths of the short-term rate and excess liquidity over the future 13 weeks. Excess liquidity (metab_13w) is the unabsorbed residual of money-supply expansion that is not absorbed by real output or prices, and its formula is given in Table 3.4.",
 46: "Table 3.3: Conditional Variables",
 47: "Table 3.4: Excess Liquidity (metab_13w) Definition",
 49: "3.3 Model Architecture: Macro-Path-Conditional Counterfactual Flow (MAC-Flow)",
 50: "As shown in Figure 1, the model consists of three components: (i) a per-step main encoder, (ii) a past-summary compressor, and (iii) a Conditional Normalizing Flow head.",
 51: "Figure 1. Model Overview",
 53: "The model takes as input a time-series tensor of length L=65 (PAST_LEN=52 + FUTURE_LEN=13) with N_CH=5 channels, and the future counterfactual scenario path is exposed to the main-encoder input at every step. The counterfactual scenario path acts as an exogenous anchor at each step, and teacher-forcing, which injects the previous step's value as the initial value of the next step, is applied to suppress the accumulation of bias in the AR rollout.",
 54: "The main encoder is an MLP that independently transforms the channel vector x[t] in R^5 at each step t. The main encoder is responsible for taking the step-wise macro conditions, while the past-summary compressor compresses the past 52-week channel sequence into a fixed vector at the origin (DIM=64).",
 55: "The Flow head applies the rational-quadratic neural spline flow (RQ-NSF) structure of Durkan et al. (2019, NeurIPS) to the 1D step-wise distribution, and the base distribution uses the skew Student-t of Hansen (1994, Int. Econ. Rev.) to explicitly absorb asymmetry and tail thickness. The hyperparameters used in this study are given in Table 3.5.",
 56: "Table 3.5: Conditional NF Head Hyperparameters",
 59: "For the Past Encoder that compresses the 52-week past sequence, four models, MLP, LSTM, Transformer, and Mamba, were examined. The MLP is a non-sequential structure that flattens the past sequence and then compresses it through a bottleneck, the simplest form of compressor that aggregates the past window into a single summary vector without sequence-specific mechanisms such as recurrence or attention. Long Short-Term Memory (LSTM; Hochreiter and Schmidhuber 1997) is a recurrent neural network structure that preserves long-term dependence through input, forget, and output gate mechanisms, absorbing the sequential dependence of the time series into a cumulative hidden state. The Transformer (Vaswani et al. 2017, NeurIPS) computes the relations among all time points in the sequence in parallel through the self-attention mechanism, and preserves temporal order information with positional encoding. Mamba (Gu and Dao 2024) is a sequence encoder based on the selective state space model (SSM), with both the linear time complexity of an RNN and the expressiveness of a Transformer.",
 60: "Figure 2. Models for Encoder",
 64: "A VAE is a generative model that compresses data into the distribution of a latent variable z and then reconstructs the data from that latent representation, and as shown in Figure 3, the Conditional VAE (Sohn et al. 2015), the comparison model in this study, is an extension that takes exogenous conditions into both encoder and decoder to perform conditional generation. CondVAE uses, as exogenous conditions, the origin-time macro variables and the future 13-week short-term-rate path, learns the latent z from the past 52-week sequence, and reconstructs the future 13-week sp_return sequence one-shot from z and the conditions. The hyperparameters used in this study are given in Table 3.6.",
 65: "Figure 3. Conditional VAE",
 67: "Table 3.6: CondVAE Hyperparameters",
 70: "A GAN is a generative model that implicitly learns the real data distribution through adversarial training of a generator and a discriminator, and this study adopts WGAN-GP (Gulrajani et al. 2017) as a comparison model, combining a gradient penalty with the critic's Wasserstein-1 distance loss. The CondGAN in this study inputs, as exogenous conditions, the origin-time macro variables and the future 13-week short-term-rate path to both the generator and the critic, and generates the 13-week sp_return sequence one-shot from the latent z and the conditions. The hyperparameters used in this study are given in Table 3.7.",
 71: "Figure 4. Conditional GAN",
 73: "Table 3.7: CondGAN (WGAN-GP) Hyperparameters",
 74: "3.5 Performance Evaluation Metrics",
 75: "As shown in Table 3.8, this study evaluates model performance along three axes. First, distribution fit measures the agreement between the step-wise conditional density and the overall distribution. Second, tail-risk reproduction measures the quantitative agreement of left-tail losses and the reliability of prediction intervals; here, together with distribution-level VaR and CVaR (order-insensitive), path-dependent loss is measured with the intra-horizon loss that reflects the order of the path. Third, distribution-shape reproduction measures the agreement of dispersion, asymmetry, and tail thickness.",
 76: "Table 3.8: Performance Evaluation Metrics (3 axes)",
 79: "All model hyperparameters in this study were pre-tuned based on a validation split within the train period of the four folds. The main hyperparameter values of each model were swept and the values minimizing the 4-fold average validation NLL/week were adopted, and stability was verified through multi-seed (5 seeds) retraining. The final hyperparameters are summarized in Table 3.9.",
 82: "4.1 Counterfactual Macro-Path Analysis",
 83: "This section analyzes the impact of injecting counterfactual paths (rate and excess liquidity) into MAC-Flow. To this end, we first confirm the adequacy of the model through out-of-sample validation on historical scenarios, and then measure risk with various counterfactual scenarios. Within each of the four folds, windows of 65 weeks (52 past weeks of macro data + 13 future weeks of rate/liquidity counterfactual conditions) roll, and from each origin 13-week equity scenarios (n=1000) are generated, measured, and evaluated.",
 84: "Validation on Historical Data",
 85: "We validated, using historical data, whether MAC-Flow produces reliable conditional distributions. The test period of each fold (financial crisis, recovery, COVID crisis, inflation tightening) is out-of-sample data not used in training, and two of the folds include the worst financial-crisis and COVID periods. Because the historical time series intertwines many factors besides the money supply and rates, we evaluated based on the coverage of the scenarios.",
 86: "Comparing the model-generated paths with the actual S&P500 weekly returns on a coverage basis, all four folds were close to the nominal levels. The 95% prediction-interval coverage ranged from 93.6% to 96.7%, and the 80% coverage from 74.4% to 82.5%, close to the nominal 80%, confirming that the model produces well-calibrated distributions even in out-of-sample historical periods.",
 87: "Table 4.1  MAC-Flow Out-of-Sample Fit (conditioned on actual macro paths)",
 89: "Comparison by Interest-Rate / Excess-Liquidity LEVEL on Historical Data",
 90: "We compared, by interest-rate and excess-liquidity LEVEL, the degree to which historical scenarios including crisis situations are reproduced. Real data make factor control impossible, so multiple factors are inevitably mixed, yet even though this is out-of-sample validation, the trends of the observed and model values were consistent in direction and order. During the financial crisis (2006-2010) and COVID (2016-2020), as excess liquidity grew, the skew of both the observed and the model values deepened, and the lower the rate the larger the intra-horizon loss, which is consistent with the co-movement of equity declines and liquidity surges that arises as risk aversion strengthens in extreme crises, as shown by Brunnermeier and Pedersen (2009). By contrast, during the tightening period (2021-2025), when liquidity was on a declining trend, the opposite phenomenon appeared, with intra-horizon loss decreasing as excess liquidity expanded and the skew of both model and observed values easing under low rates. This is consistent with the discussion of Acharya et al. (2023) that liquidity contraction in the transition from quantitative easing to tightening becomes a source of vulnerability.",
 91: "Table 4.2  Model-on-Realized Validation: Liquidity (metab) Axis",
 92: "skew / intra-horizon-loss(1%) (n).  * = n<30 (excluded)",
 93: "* Liquidity binning is based on the most recent 10 years of each fold's train data",
 94: "Table 4.3  Model-on-Realized Validation: Interest Rate (tbill) Axis",
 95: "skew / intra-horizon-loss(1%) (n).  * = n<30 (excluded).",
 96: "* tbill binning is based on the 2.5% neutral rate",
 97: "4.1.3 Path Scenario Tests",
 98: "We examined the intra-horizon loss of scenarios under counterfactual paths for excess liquidity, the rate, and their combination. As in Table 4.4, paths with the same mean and variance, namely ramp up/down and step up/down, were injected as counterfactual conditions into the four folds of this study to generate scenarios, and their intra-horizon loss was measured and compared. These are differences that a Gaussian model cannot measure, so even small differences can be meaningful results. In addition, we tested uptrend continuous/discrete paths, which have different mean and variance but the same start and end points, to check whether path differences change the risk amount even when the endpoints are the same.",
 99: "Table 4.4  Path Shape Description",
 101: "(1) Liquidity Path",
 102: "We checked whether path differences under the same mean and variance produce differences in intra-horizon loss with respect to excess liquidity. As shown in Table 4.5, the comparison shows that under the same mean and variance, a step-up liquidity path enlarges intra-horizon loss on average more than a step-down path. However, in the tightening period (2021-2025), the only tightening regime among the experimental folds, the extreme value of intra-horizon loss (CVaR 1%) was larger for the step-down path. As S. Brana and S. Prat (2016) and Acharya et al. (2023) showed, this is a result of liquidity effects differing by regime, consistent with what was found in the historical scenario analysis.",
 103: "Table 4.5  Liquidity Path Results",
 105: "(2) Interest Rate Path",
 106: "For the rate, because of the zero lower bound on negative rates, experiments that equalize mean and variance by applying symmetric up/down movements are only partially possible. In an experiment that equalized mean and variance for the tightening period, intra-horizon loss was found to differ by path. In addition, in an experiment on policy-rate paths that have different mean and variance but the same start and end points, a sharp rate hike was found to enlarge risk. This confirms the study of Bernanke and Kuttner (2005), which showed that unexpected monetary-policy actions affect expected returns.",
 107: "Table 4.6  Interest Rate Path Results",
 108: "(3) Joint Path",
 109: "We examined whether, when the excess-liquidity path and the rate path are given simultaneously, the result differs from the simple sum of the individual results. Fixing the rate path at uptrend (discrete) and applying the liquidity path as step down/up and ramp up/down, we compared the sum of the individual paths with the joint-path result.",
 110: "The comparison shows that in all folds except the recovery period (2011-2015), which was a zero-rate era, paths that applied liquidity tightening (step down, ramp down) under a rising-rate condition added to intra-horizon loss, whereas paths that simultaneously applied a liquidity-easing signal (step up, ramp up) under a rising-rate condition reduced intra-horizon loss. This is a result in which the effect is amplified or offset relative to the sum of the individual conditions, consistent with the view of S. Chavleishvili (2021) emphasizing the importance of monetary policy that accounts for the interaction among policy instruments.",
 111: "Table 4.7  Joint Path (tbill = trend-up, continuous): Intra-Horizon Loss Results",
 113: "Comparison with Other Models",
 114: "We performed comparisons against traditional models and other deep generative models on historical scenarios to compare the realism of the generated scenarios.",
 115: "4.2.1 Simple Comparison with Traditional Distribution-Estimation Models",
 116: "Traditional distribution models of the GARCH family cannot finely impose future paths as counterfactual scenarios as MAC-Flow does, so for comparison we excluded MAC-Flow's future path and used GARCH-t with the same skew-t distribution applied. As shown in Table 4.8, the comparison shows that CRPS and EMD were comparable between the two models, but in interval fit (cov80/95), skew, and so on, MAC-Flow was more stable, confirming MAC-Flow's tail-risk measurement performance.",
 117: "Table 4.8  MAC-Flow (maskall) vs GARCH-X(past)-skewt: past macro used, future path excluded (information-matched)",
 119: "4.2.2 Comparison with Path-Free Distribution-Generation Models (Conditional VAE / GAN)",
 120: "We compared MAC-Flow with the deep learning models VAE and GAN for conditional scenario generation, with the future path simplified, because the existing VAE and GAN models in finance cannot impose a specific counterfactual future path. As shown in Table 4.9, MAC-Flow was more effective on most metrics.",
 121: "Table 4.9  Conditional VAE / GAN vs MAC-Flow (baseline: 1 seed)",
 123: "Ablation Studies",
 124: "4.3.1 Past Encoder Ablation",
 125: "\tComparing MLP, LSTM, Transformer, and Mamba as the past encoder, CRPS and prediction-interval calibration (cov80/95) were nearly tied among MLP, LSTM, and Mamba, but on the skew criterion the MLP reproduced the left tail most deeply at -0.78.",
 126: "Table 4.10  Past Encoder Ablation (5 seeds x 3 folds, pooled)",
 128: "4.3.2 Feature Importance (permutation importance)",
 129: "Because this model uses each input channel over the past 52 weeks, we measured importance by shuffling each feature used as a past-52-week input channel, separately from the excess-liquidity and rate paths injected as future conditions, and taking the change in the risk quantities (intra-horizon loss, CVaR1%, skew). Equity volatility (sp_std_13w) was the most important feature, and among the features excluding it, excess liquidity (metab) followed by the short-term rate (tbill) came next. In particular, excess liquidity and the short-term rate appeared as meaningful indicators for the explanatory power of skew.",
 130: "Table 4.11  Permutation Importance",
 132: "4.3.3 Counterfactual-Condition Ablation (feature ablation)",
 133: "\tWe checked the contribution of the inputs by removing excess liquidity and tbill from the model. cov80, cov95, and CVaR1% were almost unaffected by the configuration, but removing the future macro path made the skew and the CVaR1% too shallow, confirming that conditioning on the future rate and liquidity paths is the key factor in reproducing the left tail of the equity distribution at the realized level.",
 134: "Table 4.12  Feature Ablation: NLL/week, skew, CVaR1% (vs. full model)",
 138: "As the importance of monetary policy in financial markets grows ever larger, monetary policy is an important market factor that should be analyzed in its own right rather than as part of various macro policies, but because the historical data with which to analyze it are limited, counterfactual scenario analysis is meaningful. However, in finance it is hard to find a path-dependent counterfactual scenario deep generative model. This study proposed MAC-Flow, a path-dependent counterfactual generative model, using a Normalizing Flow, a deep generative model that can express flexible distributions hard to express with volatility models such as GARCH. MAC-Flow is meaningful as a model that generates counterfactual scenarios through the paths of the short-term rate and excess liquidity, the basic elements of monetary policy, and thereby measures intra-horizon loss, a path-dependent risk.",
 139: "Through MAC-Flow, we confirmed that in crises where tail risk is amplified the co-movement of low rates and the level of excess liquidity is reproduced, and that in tightening or easing periods the effect is reversed, consistent with what various studies have shown. The counterfactual-conditioned path realized by the model also matters. Through path experiments with identical mean and variance but differing shape, and through an ablation study that removed the path, we showed that the path risk that various studies have identified is realized in the scenarios the model produces. We confirmed that paths including upward shocks in rates and liquidity enlarge equity-market tail risk even under identical mean and variance, and we reproduced the phenomenon whereby, when the rate path and the liquidity path are combined, the impact on the equity market is amplified or offset.",
 141: "Many deep learning models in finance focus on prediction, because if the predictability of financial markets can be raised it becomes a weapon that improves both portfolio profitability and risk management. However, all financial-market models cannot escape the limits of the available historical data, and the model's predictive power can be impaired by abrupt regime shifts and market events. For this reason, a scenario-based approach that responds to circumstances is an important approach in financial-market risk management.",
 142: "The central bank's future monetary-policy path is not deterministic. The U.S. Federal Reserve announces the expected rate path through the dot plot and so on, but it can change due to various factors such as the business cycle and the international situation. Liquidity is the same. Excess liquidity is one of the areas where the central bank has influence, but it is hard to predict because various macro factors are intertwined. The counterfactual scenario model proposed here makes it possible to generate scenarios according to the rate and monetary condition path for the short-term rate and liquidity, which are among the major triggers of financial markets. In particular, rather than medium- to long-term forecasting, this model generates scenarios linked to the short-term rate and liquidity paths over the 13 weeks of at most two to three FOMC meetings, and so can be a useful controlled-experiment tool for portfolio managers and analysts.",
 143: "In addition, this model can be used as a way to contribute to the data-scarcity problem, a difficulty in the machine-learning training process in finance. By assigning various rate and liquidity assumptions to historical scenarios and generating rich scenario data, one can use it for model training and simulation. However, fidelity verification and effectiveness verification to confirm the usefulness of the generated sample data must be carried out in parallel.",
 145: "This model produced counterfactual scenarios based on historical data and given rate-liquidity scenarios. The validation showed that the impact of rate and liquidity paths is amplified in extreme situations such as the financial crisis. However, the cases of extreme crises observable in the study data amount to only the financial crisis and the COVID pandemic. Because this model is a result learned from the given data, the scenarios it produces for situations and inputs beyond that data have limited meaning. The fact that, in the validation, a path experiment using mid-level rate data was possible only for the tightening period (2021-2025) in the four-fold test data is also due to this limitation.",
 146: "In addition, the historical-scenario validation of this model has the limitation that, because many factors are mixed in the real data, it is hard to separate the pure effect of rates and liquidity. Also, in some segments the number of observations is small, so the comparison of those cells is only for reference. Therefore the agreement between observed and model values should be interpreted not at the absolute level but in terms of consistency of direction and order.",
 147: "This model is not a model that predicts which rate and liquidity path will actually be realized, but a model that produces a conditional distribution given an assumed policy path. Therefore the results of this model should be interpreted not as absolute forecasts but as scenarios conditioned on a policy path.",
 149: "This study addressed only the two core elements of monetary policy, the short-term rate and excess liquidity, as counterfactual conditions. However, equity-market tail risk is also driven by various factors such as credit spreads, exchange rates, fiscal policy, and geopolitical shocks. The fact that these factors are not included as conditioning variables limits the explanatory scope of the scenarios this model produces, and extending the model to include them is a future task.",
 150: "As a controlled-experiment tool that produces tail-risk scenarios from announced expected policy paths such as the Federal Reserve's dot plot, this model allows follow-up research that integrates it into the stress testing and investment-strategy scenarios of actual portfolios. Also, although this study was limited to the U.S. S&P500, the model's applicability can be further extended to other equity markets and asset classes, longer sample periods, and additional crisis regimes.",
 151: "The MAC-Flow proposed in this study is a model that quantifies equity tail risk under unobserved monetary-policy paths, and the realism of its generated scenarios can be further raised through advanced architectures and the combination of various macro conditions. We hope that the path-dependent scenario model proposed here will develop into an analytical axis offering a different perspective from existing financial deep generative models that have focused on prediction.",
 # ---- Table 3.1 ----
 159: "Variable", 160: "Source", 161: "Original frequency", 162: "S&P500 close", 165: "3M T-bill rate",
 # ---- Table 3.2 ----
 184: "Train period", 185: "Val period", 186: "Test period",
 187: "Financial Crisis", 191: "Easing period", 195: "COVID", 199: "Tightening period",
 # ---- Table 3.3 ----
 203: "Category", 204: "Variable", 205: "Model variable", 206: "Applied window",
 207: "Hypothetical conditional path", 208: "(counterfactual injection)", 209: "Short-term rate", 211: "Future 13 weeks",
 213: "Hypothetical conditional path", 214: "(counterfactual injection)", 215: "Excess liquidity", 217: "Future 13 weeks",
 219: "Observed data", 220: "(past 52 weeks, all channels)", 221: "Equity return", 223: "Past 52 weeks",
 225: "Observed data", 226: "(past 52 weeks, all channels)", 227: "Short-term rate", 229: "Past 52 weeks",
 231: "Observed data", 232: "(past 52 weeks, all channels)", 233: "Excess liquidity", 235: "Past 52 weeks",
 237: "Observed data", 238: "(past 52 weeks, all channels)", 239: "Business conditions index", 241: "Past 52 weeks",
 243: "Observed data", 244: "(past 52 weeks, all channels)", 245: "WTI crude oil", 247: "Past 52 weeks",
 # ---- Table 3.4 ----
 249: "Item", 250: "Definition",
 252: "13-week cumulative change in money supply (M2), log%",
 254: "13-week percentage change in industrial production (INDPRO)",
 256: "13-week cumulative change in consumer prices (CPI), log%",
 # ---- Table 3.5 ----
 257: "Item", 258: "Value", 259: "Flow type", 269: "Number of blocks (transform blocks)",
 # ---- Table 3.6 ----
 271: "Item", 272: "Value",
 # ---- Table 3.7 ----
 290: "Item", 291: "Value",
 # ---- Table 3.8 ----
 306: "Category", 307: "Metric", 308: "Definition", 309: "Source",
 310: "Distribution fit", 313: "Per-week average of step-wise negative log-likelihood",
 315: "Distribution fit", 318: "Continuous Ranked Probability Score, pooled by origin",
 320: "Distribution fit", 323: "First-order Wasserstein distance between the actual and model-simulated distributions",
 325: "Tail risk", 330: "Tail risk", 333: "Hit rate of actual values within the 50% / 80% / 95% prediction intervals",
 335: "Tail risk", 338: "Maximum cumulative loss relative to entry within the holding period (order-sensitive)",
 340: "Tail risk", 343: "peak-to-trough (includes give-back, used as auxiliary)",
 345: "Distribution shape", 348: "Skewness: actual vs simulated, sign and magnitude", 352: "Excess kurtosis",
 # ---- Table 4.1 ----
 375: "Test period", 380: "Financial Crisis (2006-2010)", 385: "Recovery (2011-2015)",
 390: "COVID Crisis (2016-2020)", 395: "Inflation Tightening (2021-2025)", 400: "Average",
 # ---- Table 4.2 ----
 405: "Period", 406: "(mean/slope)", 413: "Period", 414: "(mean/slope)",
 427: "Financial Crisis", 437: "Recovery", 449: "COVID", 463: "Tightening period",
 # ---- Table 4.3 ----
 477: "Period", 478: "(mean/slope)", 485: "Period", 486: "(mean/slope)",
 499: "Financial Crisis", 513: "Recovery", 523: "COVID", 535: "Tightening period",
 # ---- Table 4.4 ----
 562: "sharp drop (start) then sharp rise (mid)", 570: "sharp rise (start) then sharp drop (mid)",
 578: "sharp drop (start) then linear rise", 586: "sharp rise (start) then linear decline",
 593: "linear rise from the start", 600: "rise with a jump at the midpoint",
 # ---- Table 4.5 ----
 602: "Shape", 606: "Financial Crisis", 612: "Financial Crisis", 618: "Recovery", 624: "Recovery",
 630: "COVID Crisis", 636: "COVID Crisis", 642: "Tightening period", 648: "Tightening period",
 # ---- Table 4.6 ----
 654: "Category", 656: "Shape",
 662: "Tightening period", 670: "Tightening period", 678: "Tightening period", 686: "Tightening period",
 695: "Financial Crisis", 704: "Financial Crisis", 713: "Recovery", 722: "Recovery",
 731: "COVID Crisis", 740: "COVID Crisis", 749: "Tightening period", 758: "Tightening period",
 # ---- Table 4.7 ----
 764: "Liquidity condition", 770: "Financial Crisis", 776: "Recovery", 782: "COVID Crisis", 788: "Tightening period",
 # ---- Table 4.8 ----
 794: "Test period", 802: "std ratio",
 803: "Financial Crisis", 813: "Financial Crisis", 823: "Recovery", 833: "Recovery",
 843: "COVID Crisis", 853: "COVID Crisis", 863: "Tightening period", 873: "Tightening period",
 883: "Average", 892: "Average",
 # ---- Table 4.9 ----
 907: "std ratio",
 908: "Financial Crisis", 916: "Financial Crisis", 924: "Financial Crisis",
 932: "Recovery", 940: "Recovery", 948: "Recovery",
 956: "COVID Crisis", 964: "COVID Crisis", 972: "COVID Crisis",
 980: "Tightening period", 988: "Tightening period", 996: "Tightening period",
 # ---- Table 4.10 ----
 1010: "std ratio",
 # ---- Table 4.11 ----
 1039: "Input variable", 1043: "Equity volatility (sp_std_13w)", 1047: "Excess liquidity (metab_13w)",
 1051: "Short-term rate (tbill_wr)", 1055: "Business conditions (ads_lag)", 1059: "Oil price (wti_wr)",
 1063: "Equity skew (sp_skew_13w)",
 # ---- Table 4.12 ----
 1068: "Configuration",
 1073: "Financial Crisis", 1080: "Financial Crisis", 1087: "Financial Crisis", 1094: "Financial Crisis",
 1075: "full (main model)", 1103: "full (main model)", 1131: "full (main model)", 1159: "full (main model)",
 1101: "Recovery", 1108: "Recovery", 1115: "Recovery", 1122: "Recovery",
 1129: "COVID Crisis", 1136: "COVID Crisis", 1143: "COVID Crisis", 1150: "COVID Crisis",
 1157: "Tightening period", 1164: "Tightening period", 1171: "Tightening period", 1178: "Tightening period",
 1173: "maskall (path removed)", 1180: "actual (realized)",
}


def has_hangul(s):
    return any('가' <= ch <= '힣' for ch in s)


def main():
    doc = docx.Document(SRC)
    paras = list(iter_paras(doc))
    applied = 0
    for i, p in enumerate(paras):
        if i in TRANS:
            set_para_text(p, TRANS[i])
            applied += 1
    doc.save(DST)
    print(f"applied {applied} translations of {len(TRANS)} entries; total paras {len(paras)}")
    # verify: any remaining Hangul?
    doc2 = docx.Document(DST)
    leftover = [(i, p.text) for i, p in enumerate(iter_paras(doc2)) if has_hangul(p.text)]
    print(f"remaining Hangul paragraphs: {len(leftover)}")
    for i, t in leftover[:60]:
        print(f"  [{i}] {t[:80]}")


if __name__ == "__main__":
    main()
