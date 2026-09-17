# Fetched 2026-09-12T00:26:35.064Z
Intent: Confirm implementation precision and mixed-signal architecture

GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs · GitHub (https://github.com/microsoft/BitNet)
citeturn13view0 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"turn4view1","lineno":185}); Total lines: 482
L35:     * BY COMPANY SIZE
L36:       * cite19†Enterprises L37:       * cite20†Small and medium teams L58:     * EXPLORE BY TOPIC
L59:       * cite34†AI L60:       * cite35†Software Development L61:       * cite36†DevOps L62:       * cite37†Security L63:       * cite38†View all topics L64: 
L65:     * EXPLORE BY TYPE
L66:       * cite39†Customer stories L67:       * cite40†Events & webinars L68:       * cite41†Ebooks & reports L69:       * cite42†Business insights L70:       * cite43†GitHub Skills†skills.github.com L71:     * SUPPORT & SERVICES
L72:       * cite14†Documentation†docs.github.com L73:       * cite44†Customer support†support.github.com L74:       * cite45†Community forum L75:       * cite46†Trust center L76:       * cite47†Partners L77: 
L78: cite48†View all resources L79: 
L80:   * Open Source
L81: 
L82:     * COMMUNITY
L83:       * cite49†GitHub SponsorsFund open source developers L84:     * PROGRAMS
L85:       * cite50†Security Lab†securitylab.github.com L86:       * cite51†Maintainer Community†maintainers.github.com L87:       * cite52†GitHub Stars†stars.github.com L88:       * cite53†Archive Program†archiveprogram.github.com L89: 
L90:     * REPOSITORIES
L91:       * cite54†Topics L92:       * cite55†Trending L93:       * cite56†Collections L94: 
L95:   * Enterprise
L96: 
L97:     * ENTERPRISE SOLUTIONS
L98:       * cite19†Enterprise platformAI-powered developer platform L99:     * AVAILABLE ADD-ONS
L100:       * cite10†GitHub Advanced SecurityEnterprise-grade security features L101:       * cite57†Copilot for BusinessEnterprise-grade AI features L102:       * cite58†Premium SupportEnterprise-grade 24/7 support L103: 
L104:   * cite59†Pricing L105: 
L106: Search`/`
L107: 
L108: cite1†Sign in L109: 
L110: cite60†Sign up L111: 
L112: Appearance settings
L113: You signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert
L114: ### Uh oh!
L115: 
L116: There was an error while loading. Please reload this page.
L117: 
L118: cite61†microsoft / cite62†BitNet Public
L119: 
L120:   * cite63†Notifications You must be signed in to change notification settings
L121:   * cite63†Fork 3.7k L122:   * cite63†Star L123: 
L124:   * cite62†Code L125:   * cite64†Issues 207 L126:   * cite65†Pull requests 120 L127:   * cite66†Discussions L128:   * cite67†Actions L129:   * cite68†Projects L130:   * cite69†Security and quality 0 L131:   * cite70†Insights L132: 
L133: Additional navigation options
L134: 
L135: main
L136: 
L137: cite71†Branches cite72†Tags L138: 
L139: [Input: Go to file]
L140: 
L141: Go to file
L142: 
L143: Code
L144: 
L145: Open more actions menu
L146: ## Latest commit
L147: 
L148: 
L149: 
L150: ## History
L151: 
L152: 110 Commits
L153: ## Folders and files
L154: Name  | Name  | Last commit message  | Last commit date
L155: --- | --- | --- | ---
L156: cite73†3rdparty | cite73†3rdparty |    |
L157: cite74†assets | cite74†assets |    |
L158: cite75†docs | cite75†docs |    |
L159: cite76†gpu | cite76†gpu |    |
L160: cite77†include | cite77†include |    |
L161: cite78†media | cite78†media |    |
L162: cite79†preset_kernels | cite79†preset_kernels |    |
L163: cite80†src | cite80†src |    |
L164: cite81†utils | cite81†utils |    |
L165: cite82†.gitignore | cite82†.gitignore |    |
L166: cite83†.gitmodules | cite83†.gitmodules |    |
L167: cite84†CMakeLists.txt | cite84†CMakeLists.txt |    |
L168: cite85†CODE_OF_CONDUCT.md | cite85†CODE_OF_CONDUCT.md |    |
L169: cite86†LICENSE | cite86†LICENSE |    |
L170: cite87†README.md | cite87†README.md |    |
L171: cite88†SECURITY.md | cite88†SECURITY.md |    |
L172: cite89†requirements.txt | cite89†requirements.txt |    |
L173: cite90†run_inference.py | cite90†run_inference.py |    |
L174: cite91†run_inference_server.py | cite91†run_inference_server.py |    |
L175: cite92†setup_env.py | cite92†setup_env.py |    |
L176: [Button: View all files]
L177: ## Repository files navigation
L178: 
L179:   *   * cite93†README L180:   * cite93†Code of conduct L181:   * cite93†MIT license L182:   * cite93†Security L183: 
L184: More items
L185: 
L186: # bitnet.cpp
L187: ### 📰 News
L188: 
L189: 07/23/2026: 📣 We released cite94†VibeASR.cpp — a real-time multilingual ASR inference engine on CPU using BitNet I2_S quantization, achieving RTF < 1 with very few threads on x86 (AVX2) and ARM (NEON) platforms. [cite94†Code ] [cite95†Models†huggingface.co ] [cite96†Report†arxiv.org ]
L190: 07/20/2026: 📣 We released cite97†BitNet-embedding-0.6B†huggingface.co and cite98†BitNet-embedding-270M†huggingface.co on Hugging Face — the first 1-bit embedding models that deliver competitive embedding quality with significantly faster inference on CPUs.
L191:   * 1.42x to 2.28x speedup over F16 on BitNet-embedding-0.6B prefill (8 threads)
L192:   * 1.32x to 1.74x speedup over F16 on BitNet-embedding-270M prefill (8 threads)
L193:   * Supports I2_S conversion with optimized kernels on x86 CPUs
L194:   * Lossless inference with 2 bits per weight
L195: 
L196: 07/16/2026: 📣 Released cite99†BitNet Embeddings 0.6B/270M: I2_S Conversion and Inference Optimization — detailed guide for converting and running BitNet embedding models with optimized I2_S kernels.
L197: 01/15/2026: 📣 Released cite100†BitNet CPU Inference Optimization — parallel kernel implementations with configurable tiling and embedding quantization support, achieving 1.15x to 2.1x additional speedup over the original implementation.
L198: 
L199: 05/20/2025: 📣 Released cite101†BitNet Official GPU inference kernel — extending 1-bit inference beyond CPUs.
L200: 
L201: 04/14/2025: 📣 Released cite102†BitNet Official 2B Parameter Model†huggingface.co on Hugging Face — the first official BitNet b1.58 model trained with 4T tokens.
L202: 02/18/2025: 📑 cite103†Bitnet.cpp: Efficient Edge Inference for Ternary LLMs†arxiv.org — system-level paper on bitnet.cpp's architecture and design.
L203: 
L204: 11/08/2024: 📑 cite104†BitNet a4.8: 4-bit Activations for 1-bit LLMs†arxiv.org — enabling 4-bit activations for further efficiency gains.
L205: 
L206: 10/21/2024: 📑 cite105†1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs†arxiv.org — the technical report behind bitnet.cpp.
L207: 
L208: 10/17/2024: 📣 bitnet.cpp 1.0 released.
L209: 03/21/2024: 📑 cite106†The-Era-of-1-bit-LLMs: Training Tips, Code, FAQ L210: 
L211: 02/27/2024: 📑 cite107†The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits†arxiv.org — the foundational paper introducing BitNet b1.58.
L212: 
L213: 10/17/2023: 📑 cite108†BitNet: Scaling 1-bit Transformers for Large Language Models†arxiv.org — the original BitNet paper.
L214: ## Overview
L215: 
L216: bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels that support fast and lossless inference of 1.58-bit models on CPU and GPU (NPU support coming next).
L217: 
L218: Try it out via this cite109†online demo†demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net , or build and run it on your own cite110†CPU or cite101†GPU .
L219: bitnet.cpp achieves speedups of 1.37x to 5.07x on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by 55.4% to 70.0%, further boosting overall efficiency. On x86 CPUs, speedups range from 2.37x to 6.17x with energy reductions between 71.9% to 82.2%.
L220: Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the cite105†technical report†arxiv.org for more details.
L221: ## Model Releases
L222: ### 1. cite102†BitNet-b1.58-2B-4T†huggingface.co - 1-bit Large Language Model
L223: 
L224: BitNet-b1.58-2B-4T is the first official BitNet b1.58 model with 2.4B parameters, trained on 4 trillion tokens. It is a ternary (1.58-bit) language model that delivers competitive performance with full-precision models of similar size while enabling significantly faster and more energy-efficient inference.
L225:   * Fast CPU Inference: Achieves up to 6.17x speedup on x86 CPUs and 5.07x on ARM CPUs compared to full-precision models.
L226:   * Energy Efficient: Reduces energy consumption by up to 82.2% on x86 and 70.0% on ARM.
L227:   * GPU Support: Official GPU inference kernel available for accelerated deployment.
L228:   * Chat-Ready: Supports conversational mode for interactive use.
L229: cite102†🤗 Hugging Face†huggingface.co | cite109†🔗 Online Demo†demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net | cite105†📄 Technical Report†arxiv.org L230: ### 2. cite97†BitNet-embedding-0.6B†huggingface.co - 1-bit Embedding Model
L231: 
L232: BitNet-embedding-0.6B is a 0.6B-parameter 1-bit embedding model that achieves competitive embedding quality with significantly faster CPU inference. It is the first model to demonstrate that ternary weights can deliver strong performance on embedding tasks.
L233:   * 1.42x to 2.28x speedup over F16 on prefill (8 threads, x86)
L234:   * Lossless Quality: Competitive embedding quality with 2 bits per weight
L235:   * I2_S Kernel: Supports optimized I2_S conversion on x86 CPUs
L236: 
L237: cite97†🤗 Hugging Face†huggingface.co | cite99†📄 I2_S Guide L238: ### 3. cite98†BitNet-embedding-270M†huggingface.co - Lightweight 1-bit Embedding Model
L239: 
L240: BitNet-embedding-270M is a compact 270M-parameter 1-bit embedding model designed for resource-constrained environments, offering fast inference with minimal memory footprint.
L241: 
L242:   * 1.32x to 1.74x speedup over F16 on prefill (8 threads, x86)
L243:   * Lossless Quality: Competitive embedding quality with 2 bits per weight
L244:   * Lightweight: Only 270M parameters for edge deployment scenarios
L245: cite98†🤗 Hugging Face†huggingface.co | cite99†📄 I2_S Guide L246: ## Supported Models
L247: Model  | Parameters  | CPU  | Kernel
L248: --- | --- | --- | ---
L249: I2_S  | TL1  | TL2
L250: --- | --- | ---
L251: Official Models
L252: ---
L253: cite102†BitNet-b1.58-2B-4T†huggingface.co | 2.4B  | x86  | ✅  | ❌  | ✅
L254: ARM  | ✅  | ✅  | ❌
L255: cite97†BitNet-embedding-0.6B†huggingface.co | 0.6B  | x86  | ✅  | ❌  | ❌
L256: ARM  | ❌  | ❌  | ❌
L257: cite98†BitNet-embedding-270M†huggingface.co | 270M  | x86  | ✅  | ❌  | ❌
L258: ARM  | ❌  | ❌  | ❌
L259: Community Models
L260: ---
L261: cite111†bitnet_b1_58-large†huggingface.co | 0.7B  | x86  | ✅  | ❌  | ✅
L262: ARM  | ✅  | ✅  | ❌
L263: cite112†bitnet_b1_58-3B†huggingface.co | 3.3B  | x86  | ❌  | ❌  | ✅
L264: ARM  | ❌  | ✅  | ❌
L265: cite113†Llama3-8B-1.58-100B-tokens†huggingface.co | 8.0B  | x86  | ✅  | ❌  | ✅
L266: ARM  | ✅  | ✅  | ❌
L267: cite114†Falcon3 Family†huggingface.co | 1B-10B  | x86  | ✅  | ❌  | ✅
L268: ARM  | ✅  | ✅  | ❌
L269: cite115†Falcon-E Family†huggingface.co | 1B-3B  | x86  | ✅  | ❌  | ✅
L270: ARM  | ✅  | ✅  | ❌
L271: ❗️We use existing 1-bit LLMs available on cite116†Hugging Face†huggingface.co to demonstrate the inference capabilities of bitnet.cpp. We hope the release of bitnet.cpp will inspire the development of 1-bit LLMs in large-scale settings in terms of model size and training tokens.
--------------------------------------------------------------------------------
BitNet b1.58 2B4T Technical Report (https://arxiv.org/html/2504.12285v1)
citeturn13view1 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"turn4view2","lineno":73}); Total lines: 513
L63: Open-source large language models (LLMs) have become pivotal in democratizing access to advanced AI capabilities, fostering innovation, and enabling research across diverse fields such as natural language processing, code generation, and vision computing (cite39†Dubey et al.,, 2024 ; cite40†Yang et al.,, 2024 ; cite41†Bai et al.,, 2025 ). Their public availability allows for widespread experimentation and adaptation.
L64: However, a significant barrier hinders their broader adoption: the substantial computational resources required for deployment and inference. State-of-the-art open LLMs typically require large memory footprints, consume considerable energy, and exhibit notable inference latency, rendering them impractical for many edge devices, resource-constrained environments, and real-time applications.
L65: 1-bit LLMs, representing an extreme yet promising form of model quantization where weights and potentially activations are constrained to binary {-1, +1} or ternary {-1, 0, +1}, offer a compelling solution to the efficiency challenges. By drastically reducing the memory required to store weights and enabling highly efficient bitwise computations, they have the potential to significantly lower deployment costs, reduce energy consumption, and accelerate inference speeds.
L66: While prior work has explored 1-bit models, existing open efforts often fall into two categories: 1) post-training quantization (PTQ) methods applied to pre-trained full-precision models, which can lead to significant performance degradation (cite42†Xu et al., 2024b, ; cite43†Team,, 2024 ), or 2) native 1-bit models (trained from scratch with 1-bit weights) that have been developed at relatively smaller scales (e.g., OLMo-Bitnet-1B^{1}^{1} 1 cite44†https://huggingface.co/NousResearch/OLMo-Bitnet-1B†huggingface.co ]) and may not yet match the capabilities of larger, full-precision counterparts.
L67: This performance gap has limited the practical impact of 1-bit LLMs thus far.
L68: To bridge this gap between efficiency and performance, we introduce BitNet b1.58 2B4T, the first open-source, native 1-bit LLM trained at scale. This model, comprising 2 billion parameters, was trained from scratch on a substantial dataset of 4 trillion tokens, leveraging architectural and training innovations specific to the 1-bit paradigm.
L69: The core contribution of this work is to demonstrate that a native 1-bit LLM, when trained effectively at scale, can achieve performance comparable to leading open-weight, full-precision models of similar size across a wide range of tasks.
L70: This technical report details the development and evaluation of BitNet b1.58 2B4T. We describe the architecture and training methodology, and then present comprehensive evaluation results on standard benchmarks assessing language understanding, mathematical reasoning, coding proficiency, and multi-turn conversational abilities. Our findings confirm its strong performance relative to established full-precision baselines, coupled with significant advantages in efficiency.
L71: Finally, we announce the public release of the BitNet b1.58 2B4T model weights via Hugging Face and provide open-source inference code optimized for both GPU and CPU execution, aiming to facilitate further research and the practical deployment of highly efficient LLMs.
L72: ## 2 Architecture
L73: 
L74: The architecture of BitNet b1.58 2B4T is derived from the standard Transformer model (cite45†Vaswani et al.,, 2017 ), incorporating significant modifications based on the BitNet framework (cite46†Wang et al., 2023a, ; cite47†Ma et al.,, 2024 ). The model is trained entirely from scratch.
L75: The core architectural innovation lies in replacing the standard full-precision linear layers (torch.nn.Linear) with custom BitLinear layers. This constitutes the foundation of the BitNet approach. Within these BitLinear layers:
L76: 
L77:   * •
L78: Weight Quantization: Model weights are quantized to 1.58 bits during the forward pass. This is achieved using an absolute mean (absmean) quantization scheme, which maps weights to ternary values $\{-1,0,+1\}$. This drastically reduces the model size and enables efficient mathematical operations.
L79: 
L80:   * •
L81: 
L82: Activation Quantization: Activations flowing through the linear projection are quantized to 8-bit integers. This employs an absolute maximum (absmax) quantization strategy, applied per-token.
L83: 
L84:   * •
L85: Normalization: We incorporate subln normalization (cite48†Wang et al.,, 2022 ) to further enhance training stability, which can be particularly beneficial in quantized training regimes.
L86: 
L87: Beyond the BitLinear layers, several established LLM techniques are integrated to enhance performance and stability:
L88: 
L89:   * •
L90: Activation Function (FFN): Within the feed-forward network (FFN) sub-layers, instead of the commonly used SwiGLU activation (cite49†Shazeer,, 2020 ), BitNet b1.58 2B4T employs squared ReLU ($\text{ReLU}^{2}$). This choice is motivated by its potential to improve model sparsity and computational characteristics within the 1-bit context (cite50†Wang et al., 2024b, ; cite51†Wang et al., 2024a, ).
L91: 
L92:   * •
L93: Positional Embeddings: Rotary Position Embeddings (RoPE) (cite52†Su et al.,, 2024 ) are used to inject positional information, a standard practice in modern high-performance LLMs.
L94: 
L95:   * •
L96: 
L97: Bias Removal: Consistent with architectures like LLaMA, all bias terms are removed from the linear layers and normalization layers throughout the network, reducing parameter count and potentially simplifying quantization.
L98: For tokenization, we adopt the tokenizer developed for LLaMA 3 (cite39†Dubey et al.,, 2024 ). This tokenizer implements a byte-level Byte-Pair Encoding (BPE) scheme with a vocabulary size of 128,256 tokens. This choice ensures robust handling of diverse text and code, and its widespread adoption facilitates straightforward integration with existing open-source tooling and ecosystems.
L99: ## 3 Training
L100: The training process for BitNet b1.58 2B4T involved three distinct phases: large-scale pre-training followed by supervised fine-tuning (SFT) and direct preference optimization (DPO).
L101: While advanced techniques like Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO) can further enhance capabilities such as mathematics and chain-of-thought reasoning (cite53†Schulman et al.,, 2017 ; cite54†Shao et al.,, 2024 ), the current version of BitNet b1.58 2B4T relies solely on pre-training, SFT, and DPO. The exploration of reinforcement learning methods remains a direction for future work.
L102: ### 3.1 Pre-training
L103: 
L104: The pre-training phase aimed to imbue the model with broad world knowledge and foundational language capabilities. We adapted general training strategies from established LLM practices (cite39†Dubey et al.,, 2024 ), with specific adjustments tailored for the 1-bit architecture.
L105: #### 3.1.1 Learning Rate Schedule
L106: 
L107: A two-stage learning rate schedule was employed.
L108: 
L109:   1. 1.
L110: 
L111: Stage 1 (High Learning Rate): The initial phase utilized a standard cosine decay schedule but commenced with a relatively high peak learning rate. This decision was informed by the observation that 1-bit models often exhibit greater training stability compared to their full-precision counterparts, allowing for more aggressive initial learning steps.
L112: 
L113:   2. 2.
L114: Stage 2 (Cooldown): Approximately midway through the planned training token count, the learning rate was abruptly decayed and subsequently maintained via a cosine schedule with a significantly lower peak value. This "cooldown" phase allows the model to refine its representations on higher-quality data (see Section cite12†3.1.3 ).
L115: #### 3.1.2 Weight Decay Schedule
L116: 
L117: Complementing the learning rate adjustments, a two-stage weight decay strategy was implemented.
L118: 
L119:   1. 1.
L120: 
L121: Stage 1: During the first training stage, weight decay followed a cosine schedule, reaching a peak value of $0.1$. This regularization helps prevent overfitting during the initial high-learning-rate phase.
L122: 
L123:   2. 2.
L124: Stage 2: In the second stage, weight decay was effectively disabled (set to zero). This allows the model parameters to settle into finer-grained optima guided by the lower learning rate and curated data.
L125: #### 3.1.3 Pre-training Data
L126: The pre-training corpus comprised a mixture of publicly available text and code datasets, including large web crawls like DCLM (cite55†Li et al., 2024b, ) and educational web pages like FineWeb-EDU (cite56†Penedo et al.,, 2024 ). To enhance mathematical reasoning abilities, we also incorporated synthetically generated mathematical data.
L127: The data presentation strategy aligned with the two-stage training: the bulk of general web data was processed during Stage 1, while higher-quality curated datasets were emphasized during the Stage 2 cooldown phase, coinciding with the reduced learning rate.
L128: ### 3.2 Supervised Fine-tuning (SFT)
L129: 
L130: Following pre-training, the model underwent supervised fine-tuning (SFT) to enhance its instruction-following capabilities and improve its performance in conversational interaction formats.
L131: #### 3.2.1 SFT Data
L132: The SFT phase utilized a diverse collection of publicly available instruction-following and conversational datasets. These included, but were not limited to, WildChat (cite57†Zhao et al.,, 2024 ), LMSYS-Chat-1M (cite58†Zheng et al.,, 2024 ), WizardLM Evol-Instruct (cite59†Xu et al., 2024a, ), and SlimOrca (cite60†Lian et al.,, 2023 ).
L133: To further bolster specific capabilities, particularly in reasoning and complex instruction adherence, we supplemented these with synthetic datasets generated using methodologies like GLAN (cite61†Li et al., 2024a, ) and MathScale (cite62†Tang et al.,, 2024 ).
L134: #### 3.2.2 Chat Template
L135: 
L136: For conversational tasks during SFT and inference, the following chat template structure was employed:
L137: 
L138:     <|begin_of_text|>System: {system_message}<|eot_id|>
L139:     User: {user_message_1}<|eot_id|>
L140:     Assistant: {assistant_message_1}<|eot_id|>
L141:     User: {user_message_2}<|eot_id|>
L142:     Assistant: {assistant_message_2}<|eot_id|>...
L143: #### 3.2.3 Optimization Details
L144: 
L145: Several optimization choices were key during SFT:
L146: 
L147:   * •
L148: 
L149: Loss Aggregation: Instead of averaging the cross-entropy loss across tokens within a batch (mean reduction), we employed summation. Empirically, we observed that summing the losses led to improved convergence and better final performance for this model.
L150: 
L151:   * •
--------------------------------------------------------------------------------
A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference for Nature Electronics - IBM Research (https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference)
citeturn13view2 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference","lineno":10}); Total lines: 102
L0: cite0†Nature Electronics L1: 
L2: Paper
L3: 
L4: 10 Aug 2023
L5: 
L6: # A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference
L7: 
L8: cite1†View publication†www.nature.com L9: ## Abstract
L10: Analogue in-memory computing (AIMC) with resistive memory devices could reduce the latency and energy consumption of deep neural network inference tasks by directly performing computations within memory. However, to achieve end-to-end improvements in latency and energy consumption, AIMC must be combined with on-chip digital operations and on-chip communication.
L11: Here we report a multicore AIMC chip designed and fabricated in 14 nm complementary metal–oxide–semiconductor technology with backend-integrated phase-change memory. The fully integrated chip features 64 AIMC cores interconnected via an on-chip communication network. It also implements the digital activation functions and additional processing involved in individual convolutional layers and long short-term memory units.
L12: With this approach, we demonstrate near-software-equivalent inference accuracy with ResNet and long short-term memory networks, while implementing all the computations associated with the weight layers and the activation functions on the chip.
L13: For 8-bit input/output matrix–vector multiplications, in the four-phase (high-precision) or one-phase (low-precision) operational read mode, the chip can achieve a maximum throughput of 16.1 or 63.1 tera-operations per second at an energy efficiency of 2.48 or 9.76 tera-operations per second per watt, respectively.
L14: ## Related
L15: 
L16: Invited talk
L17: 
L18: ### cite2†The role of material science in neuromorphic computing L19: 
L20: Valeria Bragaglia, Tommaso Stecconi, et al.
L21: 
L22: CMD 2023
L23: 
L24: Tutorial
L25: 
L26: ### cite3†In-memory Computing Approaches for Large Language Model Acceleration L27: 
L28: Manuel Le Gallo
L29: 
L30: IEDM 2025
L31: 
L32: Conference paper
L33: 
L34: ### cite4†Prospects for photonic implementations of neuromorphic devices and systems L35: 
L36: Bert J. Offrein, Jacqueline Geler-Kremer, et al.
L37: 
L38: IEDM 2020
L39: 
L40: Talk
L41: ### cite5†Effects of Crystallization on the Conductance of HfO_{x} ReRAM by In Situ TEM Method L42: 
L43: Alexandre Foucher, Baoming Wang, et al.
L44: 
L45: MRS Fall Meeting 2022
L46: 
L47: cite6†View all publications L48: 
L49:   1. cite7†Home L50:   2. ↳ cite6†Publications L51: 
L52: ## Date
L53: 
L54: 10 Aug 2023
L55: 
L56: ## Publication
L57: 
L58: cite0†Nature Electronics L59: ## Authors
L60:   * cite8†Manuel Le Gallo L61:   * cite9†Riduan Khaddam-Aljameh L62:   * cite10†Milos Stanisavljevic L63:   * cite11†Athanasios Vasilopoulos L64:   * cite12†Benedikt Kersting L65:   * cite13†Martino Dazzi L66:   * cite14†Geethan Karunaratne L67:   * cite15†Matthias Brändli L68:   * cite16†Abhairaj Singh L69:   * cite17†Silvia M. Müller L70:   * cite18†Julian Büchel L71:   * cite19†Xavier Timoneda L72:   * cite20†Vinay Joshi L73:   * cite21†Malte J. Rasch L74:   * cite22†Urs Egger L75:   * cite23†Angelo Garofalo L76:   * cite24†Anastasios Petropoulos L77:   * cite25†Theodore Antonakopoulos L78:   * cite26†Kevin Brew L79:   * cite27†Samuel Choi L80:   * cite28†Injo Ok L81:   * cite29†Timothy Philip L82:   * cite30†Victor Chan L83:   * cite31†Claire Silvestre L84:   * cite32†Ishtiaq Ahsan L85:   * cite33†Nicole Saulnier L86:   * cite34†Vijay Narayanan L87:   * cite35†Pier Andrea Francese L88:   * cite36†Evangelos Eleftheriou L89:   * cite37†Abu Sebastian L90: IBM-affiliated at time of publication
L91: ## Topics
L92: 
L93:   * cite38†AI Hardware L94: 
L95: ## Resources
L96: 
L97:   * cite1†Publication†www.nature.com L98: 
L99: ## Share
L100: 
L101:   *   *   *

