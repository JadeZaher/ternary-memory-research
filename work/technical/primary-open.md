# Fetched 2026-09-12T00:25:25.340Z
Intent: Primary sources for ternary model efficiency and memory architecture feasibility

The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (https://arxiv.org/html/2402.17764v1)
citeturn4view0 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://arxiv.org/html/2402.17764v1","lineno":null}); Total lines: 182
L0: ##### Report GitHub Issue
L1: 
L2: [Button: ×]
L3: 
L4: Title: [Input: Enter title]
L5: 
L6: Content selection saved. Describe the issue below:
L7: 
L8: Description:
L9: 
L10: [Button: Submit without GitHub] [Button: Submit in GitHub]
L11: 
L12: cite0†Back to arXiv L13: 
L14: cite1†Why HTML?†info.arxiv.org cite2†Report Issue cite3†Back to Abstract cite4†Download PDF L15:   1. cite5†Abstract L16:   2. cite6†1 The Era of 1-bit LLMs L17:   3. cite7†2 BitNet b1.58 L18:     1. cite8†Quantization Function. L19:     2. cite9†LLaMA-alike Components. L20:   4. cite10†3 Results L21:     1. cite11†Memory and Latency L22:     2. cite12†Energy L23:     3. cite13†Throughput L24:     4. cite14†Training with 2T Tokens L25:   5. cite15†4 Discussion and Future Work L26:   6. cite16†References L27: 
L28: cite17†License: arXiv.org perpetual non-exclusive license†info.arxiv.org L29: 
L30: arXiv:2402.17764v1 [cs.CL] 27 Feb 2024
L31: # The Era of 1-bit LLMs:
L32: All Large Language Models are in 1.58 Bits
L33: Shuming Ma    Hongyu Wang^{1}^{1}footnotemark: 1     Lingxiao Ma    Lei Wang    Wenhui Wang ^{†}^{†}thanks: ˜Equal contribution. $⋄$ Corresponding author. S. Ma, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, J. Xue, F. Wei are with Microsoft Research. H. Wang and R. Wang are with University of Chinese Academy of Sciences. Affiliation:     Shaohan Huang    Li Dong    Ruiping Wang    Jilong Xue    Furu Wei^{⋄} Affiliation: cite18†https://aka.ms/GeneralAI†aka.ms L34: ###### Abstract
L35: Recent research, such as BitNet [cite19†23 ], is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single parameter (or weight) of the LLM is ternary {-1, 0, 1}.
L36: It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption. More profoundly, the 1.58-bit LLM defines a new scaling law and recipe for training new generations of LLMs that are both high-performance and cost-effective.
L37: Furthermore, it enables a new computation paradigm and opens the door for designing specific hardware optimized for 1-bit LLMs.
L38: Figure 1: 1-bit LLMs (e.g., BitNet b1.58) provide a Pareto solution to reduce inference cost (latency, throughput, and energy) of LLMs while maintaining model performance. The new computation paradigm of BitNet b1.58 calls for actions to design new hardware optimized for 1-bit LLMs.
L39: ## 1 The Era of 1-bit LLMs
L40: In recent years, the field of AI has seen a rapid growth in the size and capabilities of Large Language Models (LLMs). These models have demonstrated remarkable performance in a wide range of natural language processing tasks, but their increasing size has posed challenges for deployment and raised concerns about their environmental and economic impact due to high energy consumption.
L41: One approach to address these challenges is to use post-training quantization to create low-bit models for inference [cite20†24 , cite21†5 , cite22†2 , cite23†18 ]. This technique reduces the precision of weights and activations, significantly reducing the memory and computational requirements of LLMs. The trend has been to move from 16 bits to lower bits, such as 4-bit variants [cite21†5 , cite24†9 ]. However, post-training quantization is sub-optimal, even though it is widely used in industry LLMs.
L42: Recent work on 1-bit model architectures, such as BitNet [cite19†23 ], presents a promising direction for reducing the cost of LLMs while maintaining their performance. Vanilla LLMs are in 16-bit floating values (i.e., FP16 or BF16), and the bulk of any LLMs is matrix multiplication. Therefore, the major computation cost comes from the floating-point addition and multiplication operations.
L43: In contrast, the matrix multiplication of BitNet only involves integer addition, which saves orders of energy cost for LLMs. As the fundamental limit to compute performance in many chips is power, the energy savings can also be translated into faster computation.
L44: In addition to computation, the process of transferring model parameters from DRAM to the memory of an on-chip accelerator (e.g., SRAM) can be expensive during inference. There have been attempts to enlarge SRAM to improve throughput, but this introduces significantly higher costs than DRAM. Compared to full-precision models, 1-bit LLMs have a much lower memory footprint from both a capacity and bandwidth standpoint.
L45: This can significantly reduce the cost and time of loading weights from DRAM, leading to faster and more efficient inference.
L46: In this work, we introduce a significant 1-bit LLM variant called BitNet b1.58, where every parameter is ternary, taking on values of {-1, 0, 1}. We have added an additional value of 0 to the original 1-bit BitNet, resulting in 1.58 bits in the binary system. BitNet b1.58 retains all the benefits of the original 1-bit BitNet, including its new computation paradigm, which requires almost no multiplication operations for matrix multiplication and can be highly optimized.
L47: Additionally, it has the same energy consumption as the original 1-bit BitNet and is much more efficient in terms of memory consumption, throughput and latency compared to FP16 LLM baselines. Furthermore, BitNet b1.58 offers two additional advantages. Firstly, its modeling capability is stronger due to its explicit support for feature filtering, made possible by the inclusion of 0 in the model weights, which can significantly improve the performance of 1-bit LLMs.
L48: Secondly, our experiments show that BitNet b1.58 can match full precision (i.e., FP16) baselines in terms of both perplexity and end-task performance, starting from a 3B size, when using the same configuration (e.g., model size, training tokens, etc.).
L49: ## 2 BitNet b1.58
L50: 
L51: BitNet b1.58 is based on the BitNet architecture, which is a Transformer that replaces nn.Linear with BitLinear. It is trained from scratch, with 1.58-bit weights and 8-bit activations. Compared to the original BitNet, it introduces some modifications that we summarize below.
L52: ### Quantization Function.
L53: 
L54: To constrain the weights to -1, 0, or +1, we adopt an absmean quantization function. It first scales the weight matrix by its average absolute value, and then round each value to the nearest integer among {-1, 0, +1}:
L55: 
L56:  | $$\widetilde{W}=\mathrm{RoundClip}(\frac{W}{\gamma+\epsilon},-1,1),$$  |  | (1)
L57:  | $$\mathrm{RoundClip}(x,a,b)=\max(a,\min(b,\mathrm{round}(x))),$$  |  | (2)
L58:  | $$\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|.$$  |  | (3)
L59: The quantization function for activations follows the same implementation in BitNet, except that we do not scale the activations before the non-linear functions to the range $[0,Q_{b}]$. Instead, the activations are all scaled to $[-Q_{b},Q_{b}]$ per token to get rid of the zero-point quantization. This is more convenient and simple for both implementation and system-level optimization, while introduces negligible effects to the performance in our experiments.
L60: ### LLaMA-alike Components.
L61: The architecture of LLaMA [cite25†19 , cite26†20 ] has been the de-facto backbone for open-source LLMs. To embrace the open-source community, our design of BitNet b1.58 adopts the LLaMA-alike components. Specifically, it uses RMSNorm [cite27†27 ], SwiGLU [cite28†16 ], rotary embedding [cite29†14 ], and removes all biases.
L62: In this way, BitNet b1.58 can be integrated into the popular open-source software (e.g., Huggingface, vLLM [cite30†8 ], and llama.cpp^{1}^{1} 1 cite31†https://github.com/ggerganov/llama.cpp†github.com ) with minimal efforts.
L63: Models  | Size  | Memory (GB)$\downarrow$  | Latency (ms)$\downarrow$  | PPL$\downarrow$
L64: LLaMA LLM  | 700M  | 2.08 (1.00x)  | 1.18 (1.00x)  | 12.33
L65: BitNet b1.58  | 700M  | 0.80 (2.60x)  | 0.96 (1.23x)  | 12.87
L66: LLaMA LLM  | 1.3B  | 3.34 (1.00x)  | 1.62 (1.00x)  | 11.25
L67: BitNet b1.58  | 1.3B  | 1.14 (2.93x)  | 0.97 (1.67x)  | 11.29
L68: LLaMA LLM  | 3B  | 7.89 (1.00x)  | 5.07 (1.00x)  | 10.04
L69: BitNet b1.58  | 3B  | 2.22 (3.55x)  | 1.87 (2.71x)  | 9.91
L70: BitNet b1.58  | 3.9B  | 2.38 (3.32x)  | 2.11 (2.40x)  | 9.62
L71: Table 1: Perplexity as well as the cost of BitNet b1.58 and LLaMA LLM.
L72: Models  | Size  | ARCe  | ARCc  | HS  | BQ  | OQ  | PQ  | WGe  | Avg.
L73: LLaMA LLM  | 700M  | 54.7  | 23.0  | 37.0  | 60.0  | 20.2  | 68.9  | 54.8  | 45.5
L74: BitNet b1.58  | 700M  | 51.8  | 21.4  | 35.1  | 58.2  | 20.0  | 68.1  | 55.2  | 44.3
L75: LLaMA LLM  | 1.3B  | 56.9  | 23.5  | 38.5  | 59.1  | 21.6  | 70.0  | 53.9  | 46.2
L76: BitNet b1.58  | 1.3B  | 54.9  | 24.2  | 37.7  | 56.7  | 19.6  | 68.8  | 55.8  | 45.4
L77: LLaMA LLM  | 3B  | 62.1  | 25.6  | 43.3  | 61.8  | 24.6  | 72.1  | 58.2  | 49.7
L78: BitNet b1.58  | 3B  | 61.4  | 28.3  | 42.9  | 61.5  | 26.6  | 71.5  | 59.3  | 50.2
L79: BitNet b1.58  | 3.9B  | 64.2  | 28.7  | 44.2  | 63.5  | 24.2  | 73.2  | 60.5  | 51.2
L80: Table 2: Zero-shot accuracy of BitNet b1.58 and LLaMA LLM on the end tasks.
L81: ## 3 Results
L82: We compared BitNet b1.58 to our reproduced FP16 LLaMA LLM in various sizes. To ensure a fair comparison, we pre-trained the models on the RedPajama dataset [cite32†4 ] for 100 billion tokens. We evaluated the zero-shot performance on a range of language tasks, including ARC-Easy [cite33†25 ], ARC-Challenge [cite33†25 ], Hellaswag [cite34†26 ], Winogrande [cite35†15 ], PIQA [cite36†1 ], OpenbookQA [cite37†10 ], and BoolQ [cite38†3 ]. We also reported the validation perplexity on the WikiText2 [cite39†11 ] and C4 [cite40†13 ] datasets.
L83: We compared the runtime GPU memory and latency of both LLaMA LLM and BitNet b1.58. The results were measured using the FasterTransformer^{2}^{2} 2 cite41†https://github.com/NVIDIA/FasterTransformer†github.com codebase, which is well-optimized for LLM inference latency on GPU devices. The 2-bit kernel from Ladder [cite42†22 ] is also integrated for BitNet b1.58. We reported the time per output token, as it is the major cost for inference.
L84: Table cite43†1 summarizes the perplexity and the cost for BitNet b1.58 and LLaMA LLM. It shows that BitNet b1.58 starts to match full precision LLaMA LLM at 3B model size in terms of perplexity, while being 2.71 times faster and using 3.55 times less GPU memory. In particular, BitNet b1.58 with a 3.9B model size is 2.4 times faster, consumes 3.32 times less memory, but performs significantly better than LLaMA LLM 3B.
L85: Table cite44†2 reports the detailed results of the zero-shot accuracy on the end tasks. We followed the pipeline from lm-evaluation-harness^{3}^{3} 3 cite45†https://github.com/EleutherAI/lm-evaluation-harness†github.com to perform the evaluation. The results show that the performance gap between BitNet b1.58 and LLaMA LLM narrows as the model size increases. More importantly, BitNet b1.58 can match the performance of the full precision baseline starting from a 3B size.
L86: Similar to the observation of the perplexity, the end-task results reveal that BitNet b1.58 3.9B outperforms LLaMA LLM 3B with lower memory and latency cost. This demonstrates that BitNet b1.58 is a Pareto improvement over the state-of-the-art LLM models.
L87: ### Memory and Latency
L88: 
L89: Figure 2: Decoding latency (Left) and memory consumption (Right) of BitNet b1.58 varying the model size.
L90: We further scaled up the model size to 7B, 13B, and 70B and evaluated the cost. Figure cite46†2 illustrates the trends of latency and memory, showing that the speed-up increases as the model size scales. In particular, BitNet b1.58 70B is 4.1 times faster than the LLaMA LLM baseline. This is because the time cost for nn.Linear grows with the model size. The memory consumption follows a similar trend, as the embedding remains full precision and its memory proportion is smaller for larger models.
L91: Both latency and memory were measured with a 2-bit kernel, so there is still room for optimization to further reduce the cost.
L92: ### Energy
L93: We also estimate the arithmetic operations energy consumption of both BitNet b1.58 and LLaMA LLM. We focus mainly on the calculation for matrix multiplication, since it contributes the most to the cost of LLMs. Figure cite47†3 illustrates the composition of the energy cost. The majority of BitNet b1.58 is INT8 addition calculation, while LLaMA LLM consists of both FP16 addition and FP16 multiplication.
L94: According to the energy model in [cite48†7 , cite49†28 ], BitNet b1.58 saves 71.4 times arithmetic operations energy consumption for matrix multiplication on 7nm chips. We further reported the end-to-end energy cost for models with 512 tokens. Our results show that as the model size scales, BitNet b1.58 becomes increasingly more efficient in terms of energy consumption compared to the FP16 LLaMA LLM baseline.
L95: This is due to the fact that the percentage of nn.Linear grows with the model size, while the cost from other components is smaller for larger models.
L96: Figure 3: Energy consumption of BitNet b1.58 compared to LLaMA LLM at 7nm process nodes. On the left is the components of arithmetic operations energy. On the right is the end-to-end energy cost across different model sizes.
L97: ### Throughput
L98: 
L99: Models  | Size  | Max Batch Size  | Throughput (tokens/s)
L100: LLaMA LLM  | 70B  | 16 (1.0x)  | 333 (1.0x)
L101: BitNet b1.58  | 70B  | 176 (11.0x)  | 2977 (8.9x)
L102: Table 3: Comparison of the throughput between BitNet b1.58 70B and LLaMA LLM 70B.
L103: We compare the throughput of BitNet b1.58 and LLaMA LLM with 70B parameters on two 80GB A100 cards, using pipeline parallelism [cite50†6 ] so that LLaMA LLM 70B could be run on the devices. We increased the batch size until the GPU memory limit was reached, with a sequence length of 512. Table cite51†3 shows that BitNet b1.58 70B can support up to 11 times the batch size of LLaMA LLM, resulting an 8.9 times higher throughput.
L104: BitNet b1.58 is enabling a new scaling law with respect to model performance and inference cost. As a reference, we can have the following equivalence between different model sizes in 1.58-bit and 16-bit based on the results in Figure cite46†2 and cite47†3 .
L105: 
L106:   * •
L107: 
L108: 13B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 3B FP16 LLM.
L109: 
L110:   * •
L111: 
L112: 30B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 7B FP16 LLM.
L113: 
L114:   * •
L115: 70B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 13B FP16 LLM.
L116: ### Training with 2T Tokens
L117: The number of training tokens is a crucial factor for LLMs. To test the scalability of BitNet b1.58 in terms of tokens, we trained a BitNet b1.58 model with 2T tokens following the data recipe of StableLM-3B [cite52†17 ], which is the state-of-the-art open-source 3B model. Both models were evaluated on a benchmark that consists of Winogrande [cite35†15 ], PIQA [cite36†1 ], SciQ [cite53†21 ], LAMBADA [cite54†12 ], and ARC-easy [cite33†25 ]. We reported the zero-shot accuracy in Table cite55†4 .
L118: For tasks measured with accuracy and normalized accuracy, we take the average of the two. The results of StableLM 3b at 2T tokens are taken directly from its technical report. Our findings shows that BitNet b1.58 achieves a superior performance on all end tasks, indicating that 1.58-bit LLMs also have strong generalization capabilities.
L119: Models  | Tokens  | Winogrande  | PIQA  | SciQ  | LAMBADA  | ARC-easy  | Avg.
L120: StableLM-3B  | 2T  | 64.56  | 76.93  | 90.75  | 66.09  | 67.78  | 73.22
L121: BitNet b1.58 3B  | 2T  | 66.37  | 78.40  | 91.20  | 67.63  | 68.12  | 74.34
L122: 
L123: Table 4: Comparison of BitNet b1.58 with StableLM-3B with 2T tokens.
L124: ## 4 Discussion and Future Work
L125: 
L126: 1-bit Mixture-of-Experts (MoE) LLMs
L127: Mixture-of-Experts (MoE) have proven to be a cost-effective approach for LLMs. While it significantly reduces the computation FLOPs, the high memory consumption and inter-chip communication overhead limit its deployment and application. These challenges can be addressed by 1.58-bit LLMs. Firstly, the reduced memory footprint reduces the number of devices required to deploy MoE models. Moreover, it significantly reduces the overhead of transferring activations across networks.
L128: Ultimately, there would be no overhead if the entire models could be placed on a single chip.
L129: Native Support of Long Sequence in LLMs
L130: In the era of LLMs, the ability to handle long sequence has become a critical demand. One major challenge for long sequence inference is the memory consumption introduced by the KV caches. BitNet b1.58 represents a significant step towards native support for long sequences, as it reduces the activations from 16 bits to 8 bits, allowing the context length to be doubled given the same resources.
L131: This can be further losslessly compressed to 4 bits or even lower for 1.58-bit LLMs, which we leave as future work.
L132: LLMs on Edge and Mobile
L133: The use of 1.58-bit LLMs has the potential to greatly improve the performance of language models on edge and mobile devices. These devices are often limited by their memory and computational power, which can restrict the performance and the scale of LLMs. However, the reduced memory and energy consumption of 1.58-bit LLMs allows them to be deployed on these devices, enabling a wide range of applications that were previously not possible.
L134: This can greatly enhance the capabilities of edge and mobile devices and enable new and exciting applications of LLMs. Moreover, 1.58-bit LLMs are more friendly to CPU devices, which are the main processors used in edge and mobile devices. This means that BitNet b1.58 can be efficiently executed on these devices, further improving their performance and capabilities.
L135: New Hardware for 1-bit LLMs
L136: 
L137: Recent work like Groq^{4}^{4} 4 cite56†https://groq.com/†groq.com has demonstrated promising results and great potential for building specific hardware (e.g., LPUs) for LLMs. Going one step further, we envision and call for actions to design new hardware and system specifically optimized for 1-bit LLMs, given the new computation paradigm enabled in BitNet [cite19†23 ].
L138: ## References
L139:   * [1] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. CoRR, abs/1911.11641, 2019.
L140:   * [2] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-bit quantization of large language models with guarantees. CoRR, abs/2307.13304, 2023.
L141:   * [3] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. CoRR, abs/1905.10044, 2019.
L142:   * [4] Together Computer. Redpajama: an open dataset for training large language models, 2023.
L143:   * [5] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2023.
L144:   * [6] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In Advances in Neural Information Processing Systems, pages 103–112, 2019.
L145:   * [7] Mark Horowitz. 1.1 computing’s energy problem (and what we can do about it). In 2014 IEEE International Conference on Solid-State Circuits Conference, ISSCC 2014, Digest of Technical Papers, San Francisco, CA, USA, February 9-13, 2014, pages 10–14, 2014.
L146:   * [8] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
L147:   * [9] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. AWQ: activation-aware weight quantization for LLM compression and acceleration. CoRR, abs/2306.00978, 2023.
L148:   * [10] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. CoRR, abs/1809.02789, 2018.
L149:   * [11] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.
L150:   * [12] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics, 2016.
L151:   * [13] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. CoRR, abs/1910.10683, 2019.
L152:   * [14] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
L153:   * [15] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: an adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, pages 8732–8740, 2020.
L154:   * [16] Noam Shazeer. GLU variants improve transformer. CoRR, abs/2002.05202, 2020.
L155:   * [17] Jonathan Tow, Marco Bellagente, Dakota Mahan, and Carlos Riquelme. Stablelm 3b 4e1t.
L156:   * [18] Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better LLM quantization with hadamard incoherence and lattice codebooks. CoRR, abs/2402.04396, 2024.
L157:   * [19] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: open and efficient foundation language models. CoRR, abs/2302.13971, 2023.
L158:   * [20] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, and et al. Llama 2: open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023.
L159:   * [21] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin, editors, Proceedings of the 3rd Workshop on Noisy User-generated Text, NUT@EMNLP 2017, Copenhagen, Denmark, September 7, 2017, pages 94–106. Association for Computational Linguistics, 2017.
L160:   * [22] Lei Wang, Lingxiao Ma, Shijie Cao, Ningxin Zheng, Quanlu Zhang, Jilong Xue, Ziming Miao, Ting Cao, , and Yuqing Yang. Ladder: Efficient tensor compilation on customized data format. In OSDI, 2023.
L161:   * [23] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models. CoRR, abs/2310.11453, 2023.
L162:   * [24] Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, 2023.
L163:   * [25] Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, EMNLP-IJCNLP, 2019.
L164:   * [26] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: can a machine really finish your sentence? In Proceedings of the 57th Conference of the Association for Computational Linguistics, pages 4791–4800, 2019.
--------------------------------------------------------------------------------
GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs · GitHub (https://github.com/microsoft/BitNet)
citeturn4view1 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://github.com/microsoft/BitNet","lineno":null}); Total lines: 482
--------------------------------------------------------------------------------
BitNet b1.58 2B4T Technical Report (https://arxiv.org/html/2504.12285v1)
citeturn4view2 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://arxiv.org/html/2504.12285v1","lineno":null}); Total lines: 513
--------------------------------------------------------------------------------
LLM in a flash: Efficient Large Language Model Inference with Limited Memory - ACL Anthology (https://aclanthology.org/2024.acl-long.678/)
citeturn4view3 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://aclanthology.org/2024.acl-long.678/","lineno":null}); Total lines: 294
--------------------------------------------------------------------------------
FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU (https://proceedings.mlr.press/v202/sheng23a.html)
citeturn4view4 [wordlim: 200] Crawled: today; Content type: text/html; Source: open({"ref_id":"https://proceedings.mlr.press/v202/sheng23a.html","lineno":null}); Total lines: 51
--------------------------------------------------------------------------------
Internal Error ()
citeturn4view5 [wordlim: 200] Source: open({"ref_id":"https://www.nature.com/articles/s41586-023-06337-5","lineno":null}); Total lines: 1

