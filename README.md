# DMind-2-4B

<div align="center">
  <img src="figures/dmind-ai-logo.png" width="60%" alt="DMind-2" />
</div>

<hr>

<div align="center">
    <a href="https://dmind.ai/">
    <img alt="DMind Website" src="https://img.shields.io/badge/DMind-Homepage-blue?logo=data:image/svg+xml;base64,)"/>
  </a>
  <a href="https://huggingface.co/DMindAI">
    <img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-DMind-ffd21f?color=ffd21f&logo=huggingface"/>
  </a> 
  <a href="https://x.com/dmind_ai">
    <img alt="X" src="https://img.shields.io/badge/X-@dmindai-1DA1F2?logo=x"/>
  </a> 
    <!-- <a href="https://huggingface.co/spaces/DMindAI/DMind-1">
    <img alt="Chat"
      src="https://img.shields.io/badge/🤖%20Chat-DMind-536af5?color=536af5&logoColor=white"/>
  </a> -->
  <a href="https://discord.gg/xxwmPHU3">
    <img alt="Discord"
      src="https://img.shields.io/badge/Discord-DMind-7289da?logo=discord&logoColor=white&color=7289da"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="Code License: MIT" src="https://img.shields.io/badge/Code%20License-MIT-yellow.svg"/>
  </a>
  <a href="MODEL-LICENSE">
    <img alt="Model License: Model Agreement" src="https://img.shields.io/badge/Model%20License-Model%20Agreement-yellow.svg"/>
  </a>
  
</div>

## Table of Contents
- [Introduction](#introduction)
- [1. Model Overview](#1-model-overview)
- [2. Performance Metrics](#2-performance-metrics)
- [3. Use Cases](#3-use-cases)
- [4. Quickstart](#4-quickstart)
  - [4.1 Model Downloads](#41-model-downloads)
  - [4.2 OpenRouter API](#42-openrouter-api)
  - [4.3 OpenRouter Web Chat](#43-openrouter-web-chat)
- [Privacy & Security](#privacy--security)
- [Limitations & Disclaimers](#limitations--disclaimers)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)


## 1. Model Overview

### DMind-2
DMind-2 is a series of crypto investment analysis language models designed to provide real-time, professional crypto investment consulting services for individual investors and professional institutions. Standing on the shoulders of numerous open-source pioneers, we have successfully launched two model variants through innovative post-training techniques. 

Among these, **DMind-2-4B** is specifically optimized for edge deployment, enabling users to access institutional-grade investment analysis capabilities on local devices without concerns about data privacy or network latency.

#### Model Variants(DMind-2-4B)

- **Base Model**: Qwen3-4B-Thinking-2507
- **Parameters**: 4B
- **Training Duration**: 1 month of refined post-training
- **Hardware Requirements**: Minimum 4GB VRAM, supports consumer-grade GPUs
- **Features**: Optimized for edge deployment, achieving extreme lightweight while maintaining professional analysis capabilities



### Technical Innovations:
#### 1. Domain-Adaptive Supervised Fine-Tuning (SFT)

In building DMind-2, we deeply understand the uniqueness of the crypto investment domain—it requires not only profound blockchain technical understanding but also keen financial market insights, and most importantly, the ability to perform rigorous logical reasoning among complex on-chain data and market signals. Therefore, our domain-adaptive fine-tuning strategy fully considers these requirements from the very beginning of dataset construction. We carefully curated a total of 47.6K high-quality training samples, including 27.8K crypto domain-specific data points covering comprehensive crypto investment scenarios from DeFi protocol analysis and NFT valuation models to DAO governance decisions. These data points are not simple Q&A pairs but contain complete investment logic chains, encompassing the entire reasoning process from market observation, data analysis, and risk assessment to investment recommendations.

To ensure the model maintains fundamental financial analysis capabilities while focusing on the crypto domain, we specifically incorporated 11.2K high-quality general domain data points and 8.6K pan-financial domain data points. These datasets help the model establish a solid foundation in financial theory and market analysis frameworks, enabling it to creatively apply mature methodologies from traditional finance to the emerging crypto sector. Through this multi-layered data fusion strategy, DMind-2 can act like a professional investment advisor who understands both technology and finance, providing users with comprehensive and in-depth investment analysis.

#### 2. 🔥 Core Innovation: Distribution-Preserving Chain-of-Thought Distillation (DPCD)

DMind-2's greatest technical breakthrough lies in our innovative Distribution-Preserving Chain-of-Thought Distillation method. Traditional domain fine-tuning causes catastrophic forgetting in CoT models, where the model loses reasoning coherence while gaining domain knowledge. Our DPCD method solves this through a mathematically rigorous dual-stream architecture.

##### Core Formulation

The DPCD optimization objective combines domain adaptation with reasoning preservation through the following loss function:

$$
\mathcal{L}_{\text{DPCD}} = \underbrace{\mathcal{L}_{\text{CE}}(\theta_s, \mathcal{D}_{\text{crypto}})}_{\text{Domain Learning}} + \underbrace{\lambda(t) \cdot \sum_{i=1}^{T} \alpha_i \cdot D_{\text{KL}}(P_{\theta_s}^{(i)} \| P_{\theta_t}^{(i)})}_{\text{Distribution Preservation}} + \underbrace{\beta \cdot \mathcal{L}_{\text{QS}}(\mathcal{C}_{\theta_s})}_{\text{Quality Score}}
$$

Where:

* $\theta_s$ and $\theta_t$ represent student (trainable) and teacher (frozen) model parameters.
* $P_{\theta}^{(i)}$ denotes the probability distribution at reasoning step $i$.
* $\lambda(t) = \lambda_0 \cdot (1 + \gamma \cdot \text{complexity}(x_t))$ is the dynamic weight function.
* $\alpha_i = \exp(-\delta \cdot i/T)$ implements exponential decay for later reasoning steps.
* $\mathcal{L}_{\text{QS}}$ is the quality scoring loss ensuring reasoning coherence.



##### Dynamic Weight Adjustment Mechanism

The complexity-aware weight adjustment is formulated as:

$$
\lambda(t) = \begin{cases}
\lambda_{\text{high}} \cdot \left(1 + \tanh\left(\frac{\mathcal{H}(x_t) - \mu_{\mathcal{H}}}{\sigma_{\mathcal{H}}}\right)\right) & \text{if } \mathcal{T}(x_t) \in \{\text{DeFi Analysis, Risk Assessment}\} \\
\lambda_{\text{base}} & \text{if } \mathcal{T}(x_t) \in \{\text{Market Data, Price Query}\} \\
\lambda_{\text{base}} \cdot \left(1 + \frac{\mathcal{S}(c_t)}{|\mathcal{V}_{\text{crypto}}|}\right) & \text{otherwise}
\end{cases}
$$

Where $\mathcal{H}(x_t)$ measures reasoning complexity through chain length and branching factor, $\mathcal{S}(c_t)$ counts domain-specific terms, and $|\mathcal{V}_{\text{crypto}}|$ is the crypto vocabulary size.

This mathematical framework ensures that DMind-2 maintains Qwen3's powerful reasoning capabilities while acquiring deep crypto domain expertise. The KL divergence constraint operates at each token generation step, preserving the original model's reasoning patterns. The quality scoring mechanism $\mathcal{L}_{\text{QS}}$ filters out low-quality reasoning chains, maintaining only those paths with coherence scores above threshold $\tau = 0.85$.

Through extensive experimentation, we found optimal hyperparameters: $\lambda_{\text{base}} = 0.3$, $\lambda_{\text{high}} = 0.7$, $\beta = 0.2$, and $\delta = 0.1$. This configuration achieves a 94.1% reasoning chain completeness while improving domain-specific accuracy by 23.2% over baseline fine-tuning methods.

#### 3. Reinforcement Learning from Human Feedback (RLHF) Optimization

After completing basic domain fine-tuning, we further optimize the model using the Group Relative Policy Optimization (GRPO) algorithm. GRPO offers better stability compared to traditional PPO algorithms, which is particularly important for financial domain models—we cannot tolerate dramatic performance fluctuations during optimization as this could lead to unpredictable investment advice. During the RLHF phase, we focused on addressing two key issues: professional output formatting and safety compliance.

For professional output formatting, we constructed 4.2K carefully designed professional format data points. These data samples are sourced from real investment research reports, market analysis documents, and project due diligence reports, covering all aspects of investment analysis. Through RLHF training, the model learned how to organize a professional investment analysis report: starting with an executive summary that clearly articulates investment opportunities and risks; conducting in-depth technical analysis and market evaluation in the main body; and finally providing clear investment recommendations and risk warnings. This structured output not only improves information readability but more importantly helps investors establish systematic analytical frameworks, avoiding impulsive investment decisions due to disorganized information.

Safety alignment is another aspect we particularly emphasize. The crypto investment field is full of high-risk, high-reward opportunities, and the model must accurately identify and highlight potential risks. We use proprietary risk case datasets to conduct safety training on the model, ensuring it won't output overly optimistic investment advice or overlook obvious risk signals. For example, when analyzing an emerging DeFi protocol, the model automatically checks key risk indicators such as smart contract audit status, team background, and total value locked, explicitly marking risk levels in investment recommendations. This responsible output approach not only protects users' asset security but also reflects our commitment to financial compliance.


## 2. Performance Metrics

<div align="center">
  <img src="figures/dmind-2-performance.jpeg" width="80%" alt="DMind-2" />
</div>


| Category | Benchmark (Metric) | DeepSeek-R1-0528-Qwen3-8B | gpt-oss-20b | Qwen3-32B | Qwen3-4B(Thinking) | DMind-2-4B(4B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **General** | | | | | | |
| | MMLU-Pro (EM) | 84.0 | 85.3 | 65.54 | 70.4 | 82.1 |
| | GPQA-Diamond (Pass@1) | 71.5 | 71.5 | 68.4 | 55.9 | 64.2 |
| | SimpleQA (Correct) | 30.1 | 6.7 | 8.0 | 1.0 | - |
| **Math** | | | | | | |
| | AIME 2024 (Pass@1) | 79.8 | 96 | 81.4 | - | 92.7 |
| | AIME 2025 (Pass@1) | 70.0 | 98.7 | 72.9 | 65.6 | 81.6 |
| | CNMO 2024 (Pass@1) | 78.8 | 86.9 | - | - | 82.4 |
| **Tools** | | | | | | |
| | BFCL_v3 | - | 67.8 | 70.3 | 65.9 | 70.2 |
| **Crypto** | | | | | | |
| | DMind Benchmark | 71.5 | 70.2 | 68.0 | 74.5 | 80.1 |


## 3. Use Cases

### 🎯 Edge-Side Crypto Investment Decision Support

DMind-2 can provide real-time crypto investment analysis on users' personal devices, including DeFi yield comparisons, liquidity mining strategy optimization, and NFT valuation analysis. All calculations and analyses are completed locally, ensuring absolute privacy of investment strategies and position information. The model can analyze on-chain data, evaluate project fundamentals, identify market trends, and provide comprehensive support for investment decisions.

### 💼 Personalized Financial Advisory Services

Based on users' risk preferences, investment objectives, and asset allocation needs, DMind-2 can provide customized investment advice. Whether for long-term value investing or short-term arbitrage opportunities, the model can provide professional analysis and recommendations. More importantly, it can explain complex crypto concepts in plain language, helping investors understand the logic behind every investment decision.

### 📊 Comprehensive Financial Investment Computational Analysis

DMind-2 is not limited to the crypto domain but also possesses powerful pan-financial computational analysis capabilities. It can perform yield calculations, risk assessments, portfolio optimization, correlation analysis, and other professional financial computations. By integrating traditional financial theory with crypto innovative mechanisms, the model helps investors find optimal asset allocation solutions between old and new financial systems.

### 🔍 Real-Time Market Monitoring and Alerts

Edge-deployed DMind-2 can monitor market dynamics 24/7, promptly alerting users when important market events or investment opportunities arise. Running locally ensures extremely fast response speeds, providing immediate response recommendations during severe market volatility.


## 4. Quickstart

### 4.1 Model Downloads

| **Model**      | **Base Model** | **Download**                                                                 |
|:--------------:|:--------------:|:----------------------------------------------------------------------------:|
| DMind-2-4B        | Qwen/Qwen3-4B-Thinking-2507     | [Hugging Face Link](https://huggingface.co/DMindAI/DMind-2-4B)            |


### 4.2 OpenRouter API (Coming Soon)
*Documentation for API access will be available soon.*

### 4.3 OpenRouter Web Chat (Coming Soon)
*Web chat interface documentation will be available soon.*


## Privacy & Security

- 🔐 **Fully Localized**: All inference computations are completed on user devices, no internet required
- 🛡️ **Data Privacy**: Investment strategies and personal information never leave local devices
- ⚡ **Real-Time Response**: No network latency, millisecond-level response speed
- 🔒 **Security Compliance**: Built-in risk warning mechanisms, compliant with financial regulations


## Limitations & Disclaimers

1. **Not Investment Advice**: Model outputs are for reference only; final investment decisions require users' own judgment
2. **Market Risk**: Crypto markets are highly volatile; please carefully assess risk tolerance
3. **Knowledge Timeliness**: Model knowledge has temporal limitations; latest market information requires additional verification
4. **Regulatory Compliance**: Please comply with financial regulations in your jurisdiction when using



## Acknowledgments

We thank the Qwen and zai teams for providing the excellent base model and the continuous contributions from the open-source community. DMind-2's success wouldn't be possible without the collective efforts of the entire AI and Crypto community.


## License
- This model follows the Apache 2.0 open-source license. Commercial use must comply with relevant terms.

- **Base Models:**
  - DMind-2-4B is derived from Qwen/Qwen3-4B-Thinking-2507, originally licensed under the [Qwen License](https://github.com/QwenLM/Qwen3).
- Please ensure compliance with the original base model licenses when using or distributing derivatives.


## Contact
For questions or support, please contact team@dmind.ai

- 🌐 Project Homepage: [https://dmind.ai](https://dmind.ai)
- 💬 Community Discussion: [Discord](https://discord.gg/dmind)
- 🐦 Twitter: [@DMindAI](https://twitter.com/DMindAI)
