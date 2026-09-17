# Fetched 2026-09-12T00:25:45.015Z
Intent: Focus technical evidence and primary-author limitations

FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU (https://proceedings.mlr.press/v202/sheng23a.html)
citeturn8view0 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"turn4view4","lineno":10}); Total lines: 51
L0: Proceedings of Machine Learning Research
L1: 
L2: [Input]
L3: 
L4: cite0†Volume 202 cite1†JMLR†www.jmlr.org cite2†DMLR†data.mlr.press cite3†TMLR†jmlr.org cite4†MLOSS†www.jmlr.org cite5†FAQ cite6†Submission Format L5: 
L6: [cite7†edit†github.com ]
L7: # FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
L8: 
L9: Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Re, Ion Stoica, Ce Zhang
L10: 
L11: Proceedings of the 40th International Conference on Machine Learning, PMLR 202:31094-31116, 2023.
L12: #### Abstract
L13: The high computational and memory requirements of large language model (LLM) inference make it feasible only with multiple high-end accelerators. Motivated by the emerging demand for latency-insensitive tasks with batched processing, this paper initiates the study of high-throughput LLM inference using limited resources, such as a single commodity GPU. We present FlexGen, a high-throughput generation engine for running LLMs with limited GPU memory.
L14: FlexGen can be flexibly configured under various hardware resource constraints by aggregating memory and computation from the GPU, CPU, and disk. By solving a linear programming problem, it searches for efficient patterns to store and access tensors. FlexGen further compresses the weights and the attention cache to 4 bits with negligible accuracy loss. These techniques enable FlexGen to have a larger space of batch size choices and thus significantly increase maximum throughput.
L15: As a result, when running OPT-175B on a single 16GB GPU, FlexGen achieves significantly higher throughput compared to state-of-the-art offloading systems, reaching a generation throughput of 1 token/s for the first time with an effective batch size of 144. On the HELM benchmark, FlexGen can benchmark a 30B model with a 16GB GPU on 7 representative sub-scenarios in 21 hours. The code is available at https://github.com/FMInference/FlexGen.
L16: #### Cite this Paper
L17: 
L18: * * *
L19: 
L20: BibTeX
L21: `@InProceedings{pmlr-v202-sheng23a, title = {{F}lex{G}en: High-Throughput Generative Inference of Large Language Models with a Single {GPU}}, author = {Sheng, Ying and Zheng, Lianmin and Yuan, Binhang and Li, Zhuohan and Ryabinin, Max and Chen, Beidi and Liang, Percy and Re, Christopher and Stoica, Ion and Zhang, Ce}, booktitle = {Proceedings of the 40th International Conference on Machine Learning}, pages = {31094--31116}, year = {2023}, editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan}, volume = {202}, series = {Proceedings of Machine Learning Research}, month = {23--29 Jul}, publisher = {PMLR}, pdf = {https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf}, url = {https://proceedings.mlr.press/v202/sheng23a.html}, abstract = {The high computational and memory requirements of large language model (LLM) inference make it feasible only with multiple high-end accelerators.
L22: Motivated by the emerging demand for latency-insensitive tasks with batched processing, this paper initiates the study of high-throughput LLM inference using limited resources, such as a single commodity GPU. We present FlexGen, a high-throughput generation engine for running LLMs with limited GPU memory. FlexGen can be flexibly configured under various hardware resource constraints by aggregating memory and computation from the GPU, CPU, and disk.
L23: By solving a linear programming problem, it searches for efficient patterns to store and access tensors. FlexGen further compresses the weights and the attention cache to 4 bits with negligible accuracy loss. These techniques enable FlexGen to have a larger space of batch size choices and thus significantly increase maximum throughput.
--------------------------------------------------------------------------------
A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference for Nature Electronics - IBM Research (https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference)
citeturn8view1 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference","lineno":null}); Total lines: 102
--------------------------------------------------------------------------------
GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs · GitHub (https://github.com/microsoft/BitNet)
citeturn8view2 [wordlim: 200] Crawled: today; Content type: text/html; Source: find({"ref_id":"turn4view1","pattern":"What's New"}); Total lines: 482
No matching text found for "What's New"--------------------------------------------------------------------------------
BitNet b1.58 2B4T Technical Report (https://arxiv.org/html/2504.12285v1)
citeturn8view3 [wordlim: 200] Crawled: today; Content type: text/html; Source: find({"ref_id":"turn4view2","pattern":"Training"}); Total lines: 513
L70: This technical report details the development and evaluation of BitNet b1.58 2B4T. We describe the architecture and training methodology, and then present comprehensive evaluation results on standard benchmarks assessing language understanding, mathematical reasoning, coding proficiency, and multi-turn conversational abilities. Our findings confirm its strong performance relative to established full-precision baselines, coupled with significant advantages in efficiency.
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
L152: Hyperparameter Tuning: Careful tuning of the learning rate and the number of training epochs was performed. Consistent with our pre-training findings, the 1-bit model benefited from a relatively larger learning rate during SFT compared to typical full-precision model fine-tuning. Furthermore, achieving optimal convergence required extending the fine-tuning duration over a larger number of epochs than full-precision models of similar size.
L153: ### 3.3 Direct Preference Optimization (DPO)
L154: To further align the model’s behavior with human preferences regarding helpfulness and safety, we applied Direct Preference Optimization (DPO) (cite63†Rafailov et al.,, 2023 ) following the SFT phase. DPO offers an efficient alternative to traditional RLHF by directly optimizing the language model using preference data, thereby circumventing the need to train a separate reward model.
L155: This DPO stage served to refine the model’s conversational prowess and overall alignment with desired interaction patterns in practical use cases.
L156: #### 3.3.1 Training Data
L157: 
L158: The preference dataset used for DPO training was constructed from a combination of publicly available resources recognized for capturing diverse human judgments on model outputs. Specifically, we utilized UltraFeedback (cite64†Cui et al.,, 2024 ) and MagPie (cite65†Xu et al., 2024c, ). The aggregation of these datasets provided a robust and multifaceted preference signal, guiding the model towards generating responses more aligned with human expectations.
L159: #### 3.3.2 Training Details
L160: The DPO training phase was conducted for 2 epochs. We employed a learning rate of $2\times 10^{-7}$ and set the DPO beta parameter, which controls the divergence from the reference policy, to 0.1. To enhance training efficiency during this phase, we integrated optimized kernels from the Liger Kernel library (cite66†Hsu et al.,, 2024 ).
L161: Qualitatively, our observations indicate that the DPO process effectively steered the model towards preferred response styles without inducing significant degradation in the core capabilities established during pre-training and SFT.
L162: Benchmark (Metric)  | LLaMA 3.2  | Gemma-3  | Qwen2.5  | SmolLM2  | MiniCPM  | BitNet b1.58
L163: 1B  | 1B  | 1.5B  | 1.7B  | 2B  | 2B
L164: Memory  | 2GB  | 1.4GB  | 2.6GB  | 3.2GB  | 4.8GB  | 0.4GB
L165: (Non-emb)
L166: ---
L167: Latency  | 48ms  | 41ms  | 65ms  | 67ms  | 124ms  | 29ms
L168: (CPU; TPOT)
L169: ---
L170: Energy  | 0.258J  | 0.186J  | 0.347J  | 0.425J  | 0.649J  | 0.028J
L171: (Estimated)
L172: ---
L173: Training Tokens  | 9T  | 2T  | 18T  | 11T  | 1.1T  | 4T
L174: (Pre-training)  | (pruning & distillation)  | (distillation)
L175: ARC-Challange  | 37.80  | 38.40  | 46.67  | 43.52  | 44.80  | 49.91
L176: (0-shot; Acc,norm)
L177: ---
L178: ARC-Easy  | 63.17  | 63.13  | 76.01  | 62.92  | 72.14  | 74.79
L179: (0-shot; Acc,norm)
L180: ---
L181: OpenbookQA  | 34.80  | 38.80  | 40.80  | 46.00  | 40.20  | 41.60
L182: (0-shot; Acc,norm)
L183: ---
L184: BoolQ  | 64.65  | 74.22  | 78.04  | 75.78  | 80.67  | 80.18
L185: (0-shot; Acc)
L186: ---
L187: HellaSwag  | 60.80  | 57.69  | 68.28  | 71.71  | 70.81  | 68.44
L188: (0-shot; Acc,norm)
L189: ---
L190: PIQA  | 74.21  | 71.93  | 76.12  | 76.12  | 76.66  | 77.09
L191: (0-shot; Acc,norm)
L192: ---
L193: WinoGrande  | 59.51  | 58.48  | 62.83  | 68.98  | 61.80  | 71.90
L194: (0-shot; Acc)
L195: ---
L196: CommonsenseQA  | 58.48  | 42.10  | 76.41  | 63.55  | 71.74  | 71.58
L197: (10-shot; Acc)
L198: ---
L199: TruthfulQA  | 43.80  | 38.66  | 46.67  | 39.90  | 41.41  | 45.31
L200: (10-shot; MC2)
L201: ---
L202: TriviaQA  | 37.60  | 23.49  | 38.37  | 45.97  | 34.13  | 33.57
L203: (5-shot; EM)
L204: ---
L205: MMLU  | 45.58  | 39.91  | 60.25  | 49.24  | 51.82  | 53.17
L206: (5-shot; Acc)
L207: ---
L208: HumanEval+  | 31.10  | 37.20  | 50.60  | 28.00  | 43.90  | 38.40
L209: (0-shot; Pass@1)
L210: ---
L211: GSM8K  | 38.21  | 31.16  | 56.79  | 45.11  | 4.40  | 58.38
L212: (4-shot; EM)
L213: ---
L214: MATH-500  | 23.00  | 42.00  | 53.00  | 17.60  | 14.80  | 43.40
L215: (0-shot; EM)
L216: ---
L217: IFEval  | 62.71  | 66.67  | 50.12  | 57.91  | 36.81  | 53.48
L218: (0-shot; Instruct-Strict)
L219: ---
L220: MT-bench  | 5.43  | 6.40  | 6.12  | 5.50  | 6.57  | 5.85
L221: (0-shot; Average)
L222: ---
L223: Average  | 44.90  | 43.74  | 55.23  | 48.70  | 42.05  | 54.19
L224: Table 1: Comparison of BitNet b1.58 2B4T with leading open-weight full-precision LLMs of similar size (1B-2B parameters) on efficiency metrics and performance across a wide range of benchmarks. All models compared are instruction-tuned versions.
L225: ## 4 Evaluation
L226: Benchmark (Metric)  | Qwen2.5  | BitNet b1.58
L227: 1.5B-bf16  | 1.5B-GPTQ-int4  | 1.5B-AWQ-int4  | 2B
L228: Memory  | 2.6GB  | 0.7GB  | 0.7GB  | 0.4GB
L229: (Non-emb)
L230: ---
L231: Activation  | bf16  | bf16  | bf16  | int8
L232: MMLU  | 60.25  | 58.06  | 57.43  | 53.17
L233: (5-shot; Acc)
L234: ---
L235: GSM8K  | 56.79  | 50.57  | 50.64  | 58.38
L236: (4-shot; EM)
L237: ---
L238: IFEval  | 50.12  | 47.84  | 45.44  | 53.48
L239: (0-shot; Instruct-Strict)
L240: ---
L241: Average  | 55.72  | 52.15  | 51.17  | 55.01
L242: Table 2: Comparison of BitNet b1.58 (2B) against Qwen2.5 1.5B in its original bf16 precision and after INT4 post-training quantization (GPTQ and AWQ). All models shown are based on instruction-tuned checkpoints.
L243: Benchmark (Metric)  | Bonsai  | OLMo-Bitnet  | Falcon3-1.58bit  | Llama3-8B-1.58  | BitNet b1.58
L244: --- | --- | --- | --- | --- | ---
L245: 0.5B  | 1B  | 7B  | 8B  | 2B
L246: --- | --- | --- | --- | ---
L247: Native 1-bit  | ✓  | ✓  | ✗  | ✗  | ✓
L248: --- | --- | --- | --- | --- | ---
L249: ARC-Challange  | 33.19  | 26.54  | 37.80  | 43.69  | 49.91
L250: (0-shot; Acc,norm)
L251: ---
L252: ARC-Easy  | 58.25  | 25.38  | 65.03  | 70.71  | 74.79
L253: (0-shot; Acc,norm)
L254: ---
L255: OpenbookQA  | 33.60  | 28.20  | 38.20  | 37.20  | 41.60
L256: (0-shot; Acc,norm)
L257: ---
L258: BoolQ  | 58.44  | 52.48  | 72.14  | 68.38  | 80.18
L259: (0-shot; Acc)
L260: ---
L283: Table 3: Performance comparison of BitNet b1.58 2B4T against other open-weight 1-bit models. This includes natively trained 1-bit models (Bonsai-0.5B, OLMo-Bitnet-1B) and larger models post-training quantized to 1.58-bit (Falcon3-1.58bit-7B, Llama3-8B-1.58).
L311: ### 4.2 Comparison with Post-training Quantized Models
L312: 
L313: We further investigate the efficiency-performance trade-off by comparing BitNet b1.58 2B4T against post-training quantized (PTQ) versions of a leading competitor, Qwen2.5 1.5B, using standard INT4 methods (GPTQ and AWQ). The results are summarized in Table cite88†2 .
L320: This highlights the effectiveness of the native training approach employed by BitNet b1.58 2B4T, allowing it to set a new state-of-the-art performance level for models operating at this extreme level of quantization, even surpassing larger models subjected to post-training quantization.
--------------------------------------------------------------------------------
LLM in a flash: Efficient Large Language Model Inference with Limited Memory - ACL Anthology (https://aclanthology.org/2024.acl-long.678/)
citeturn8view4 [wordlim: 200] Crawled: today; Content type: text/html; Source: find({"ref_id":"turn4view3","pattern":"Abstract"}); Total lines: 294
L22:     * cite14†Anthology identifiers L23:     * cite15†Names L24:     * cite16†ORCID iDs L25:     * cite17†DOIs L26:     * cite18†Verified authors L27:   * cite0†Contributions L28:     * cite19†Submissions L29:     * cite20†Corrections L30:     * cite21†Maintain author pages L31:     * cite22†Attachments L32: ## cite23†LLM in a flash: Efficient Large Language Model Inference with Limited Memory L33: 
L34: cite24†Keivan Alizadeh , cite25†Seyed Iman Mirzadeh , cite26†Dmitry Belenko , cite27†S. Khatamifard , cite28†Minsik Cho , cite29†Carlo C Del Mundo , cite30†Mohammad Rastegari , cite31†Mehrdad Farajtabar L35: ##### Correct Metadata for
L36: 
L37: Use this form to create a GitHub issue with structured data describing the correction. You will need a GitHub account. Once you create that issue, the correction will be reviewed by a staff member.
L38: 
L39: ⚠️ Mobile Users: Submitting this form to create a new issue will only work with github.com, not the GitHub Mobile app.
L40: Important: The Anthology treat PDFs as authoritative. Please use this form only to correct data that is out of line with the PDF. See cite20†our corrections guidelines if you need to change the PDF.
L41: 
L42: Title Adjust the title. Retain tags such as <fixed-case>. [Input]
L43: 
L44: Authors Adjust author names and order to match the PDF.
L45: 
L46: [Button: Add Author]
L47: Abstract Correct abstract if needed. Retain XML formatting tags such as <tex-math>. You may use <b>...</b> for bold, <i>...</i> for italic, <u>...</u> for underline, <sc>...</sc> for small-caps, <tt>...<tt> for `typewriter text`, <url>...</url> for URLs, <a href=...> for hyperlinks, and <par/> for paragraph breaks.
L48: 
L49: Verification against PDF Ensure that the new title/authors match the snapshot below. (If there is no snapshot or it is too small, consult cite0†the PDF .)
L50: Authors concatenated from the text boxes above:
L51: 
L52: [Input] ALL author names match the snapshot above—including middle initials, hyphens, and accents.
L53: 
L54: [Button: Create GitHub issue for staff review]
L55: 
L56: * * *
L57: ##### Abstract
L58: Large language models (LLMs) are central to modern natural language processing, delivering exceptional performance in various tasks. However, their substantial computational and memory requirements present challenges, especially for devices with limited DRAM capacity. This paper tackles the challenge of efficiently running LLMs that exceed the available DRAM capacity by storing the model parameters in flash memory, but bringing them on demand to DRAM.
L59: Our method involves constructing an inference cost model that takes into account the characteristics of flash memory, guiding us to optimize in two critical areas: reducing the volume of data transferred from flash and reading data in larger, more contiguous chunks. Within this hardware-informed framework, we introduce two principal techniques.
L60: First, “windowing” strategically reduces data transfer by reusing previously activated neurons, and second, “row-column bundling”, tailored to the sequential data access strengths of flash memory, increases the size of data chunks read from flash memory. These methods collectively enable running models up to twice the size of the available DRAM, with a 4-5x and 20-25x increase in inference speed compared to naive loading approaches in CPU and GPU, respectively.
L61: Our integration of sparsity awareness, context-adaptive loading, and a hardware-oriented design paves the way for effective inference of LLMs on devices with limited memory.
L62: Anthology ID:
L63:     2024.acl-long.678
L64: Volume:
L65:     cite32†Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) L66: Month:
L67:     August
L68: Year:
L69:     2024
L70: Address:
L71:     Bangkok, Thailand
L72: Editors:
L73:     cite33†Lun-Wei Ku , cite34†Andre Martins , cite35†Vivek Srikumar L74: Venue:
L75:     cite36†ACL L76: SIG:
L77: 
L78: Publisher:
L79:     Association for Computational Linguistics
L80: Note:
L81: 
L82: Pages:
L83:     12562–12584
L84: Language:
L85: 
L86: URL:
L87:     cite37†https://aclanthology.org/2024.acl-long.678/ L88: DOI:
L89:     cite38†10.18653/v1/2024.acl-long.678†doi.org L90: Bibkey:
L91: 
L92: Cite (ACL):
L93:     Keivan Alizadeh, Seyed Iman Mirzadeh, Dmitry Belenko, S. Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. 2024. cite37†LLM in a flash: Efficient Large Language Model Inference with Limited Memory . In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12562–12584, Bangkok, Thailand. Association for Computational Linguistics.
L94: Cite (Informal):
L95:     cite37†LLM in a flash: Efficient Large Language Model Inference with Limited Memory (Alizadeh et al., ACL 2024)
L96: Copy Citation:
L97:     [Button: More options…]
L98: PDF:
L99:     cite23†https://aclanthology.org/2024.acl-long.678.pdf L100: cite23†PDF cite0†Cite cite0†Fix data L101: 
L102: * * *
L103: ##### Export citation
L104: 
L105:     @inproceedings{alizadeh-etal-2024-llm,
L106:         title = "{LLM} in a flash: Efficient Large Language Model Inference with Limited Memory",
L107:         author = "Alizadeh, Keivan  and
L108:           Mirzadeh, Seyed Iman  and
L109:           Belenko, Dmitry  and
L110:           Khatamifard, S.  and
L111:           Cho, Minsik  and
L112:           Del Mundo, Carlo C  and
L113:           Rastegari, Mohammad  and
L114:           Farajtabar, Mehrdad",
L115:         editor = "Ku, Lun-Wei  and
L116:           Martins, Andre  and
L117:           Srikumar, Vivek",
L118:         booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
L119:         month = aug,
L120:         year = "2024",
L121:         address = "Bangkok, Thailand",
L122:         publisher = "Association for Computational Linguistics",
L123:         url = "https://aclanthology.org/2024.acl-long.678/",
L124:         doi = "10.18653/v1/2024.acl-long.678",
L125:         pages = "12562--12584",
L126:         abstract = "Large language models (LLMs) are central to modern natural language processing, delivering exceptional performance in various tasks. However, their substantial computational and memory requirements present challenges, especially for devices with limited DRAM capacity. This paper tackles the challenge of efficiently running LLMs that exceed the available DRAM capacity by storing the model parameters in flash memory, but bringing them on demand to DRAM.

