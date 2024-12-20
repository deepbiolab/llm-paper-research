# Improving Language Understanding by Generative Pre-Training  

Alec Radford Karthik Narasimhan Tim Salimans Ilya Sutskever OpenAI OpenAI OpenAI OpenAI alec@openai.com karthikn@openai.com tim@openai.com ilyasu@openai.com  

# Abstract  

Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classiﬁcation. Although large unlabeled text corpora are abundant, labeled data for learning these speciﬁc tasks is scarce, making it challenging for disc rim i natively trained models to perform adequately. We demonstrate that large gains on these tasks can be realized by  generative pre-training  of a language model on a diverse corpus of unlabeled text, followed by  discriminative ﬁne-tuning  on each speciﬁc task. In contrast to previous approaches, we make use of task-aware input transformations during ﬁne-tuning to achieve effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our approach on a wide range of benchmarks for natural language understanding. Our general task-agnostic model outperforms disc rim i natively trained models that use architectures speciﬁcally crafted for each task, signiﬁcantly improving upon the state of the art in 9 out of the 12 tasks studied. For instance, we achieve absolute improvements of   $8.9\%$   on commonsense reasoning (Stories Cloze Test),  $5.7\%$   on question answering (RACE), and   $1.5\%$   on textual entailment (MultiNLI).  

# 1 Introduction  

The ability to learn effectively from raw text is crucial to alleviating the dependence on supervised learning in natural language processing (NLP). Most deep learning methods require substantial amounts of manually labeled data, which restricts their applicability in many domains that suffer from a dearth of annotated resources [ 61 ]. In these situations, models that can leverage linguistic information from unlabeled data provide a valuable alternative to gathering more annotation, which can be time-consuming and expensive. Further, even in cases where considerable supervision is available, learning good representations in an unsupervised fashion can provide a signiﬁcant performance boost. The most compelling evidence for this so far has been the extensive use of pre- trained word embeddings [ 10 ,  39 ,  42 ] to improve performance on a range of NLP tasks [ 8 ,  11 ,  26 ,  45 ].  

Leveraging more than word-level information from unlabeled text, however, is challenging for two main reasons. First, it is unclear what type of optimization objectives are most effective at learning text representations that are useful for transfer. Recent research has looked at various objectives such as language modeling [ 44 ], machine translation [ 38 ], and discourse coherence [ 22 ], with each method outperforming the others on different tasks.   Second, there is no consensus on the most effective way to transfer these learned representations to the target task. Existing techniques involve a combination of making task-speciﬁc changes to the model architecture [ 43 ,  44 ], using intricate learning schemes [ 21 ] and adding auxiliary learning objectives [ 50 ]. These uncertainties have made it difﬁcult to develop effective semi-supervised learning approaches for language processing.  

In this paper, we explore a semi-supervised approach for language understanding tasks using a combination of unsupervised pre-training and supervised ﬁne-tuning. Our goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. We assume access to a large corpus of unlabeled text and several datasets with manually annotated training examples (target tasks). Our setup does not require these target tasks to be in the same domain as the unlabeled corpus. We employ a two-stage training procedure. First, we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt these parameters to a target task using the corresponding supervised objective.  

For our model architecture, we use the  Transformer  [ 62 ], which has been shown to perform strongly on various tasks such as machine translation [ 62 ], document generation [ 34 ], and syntactic parsing [ 29 ]. This model choice provides us with a more structured memory for handling long-term dependencies in text, compared to alternatives like recurrent networks, resulting in robust transfer performance across diverse tasks. During transfer, we utilize task-speciﬁc input adaptations derived from traversal-style approaches [ 52 ], which process structured text input as a single contiguous sequence of tokens. As we demonstrate in our experiments, these adaptations enable us to ﬁne-tune effectively with minimal changes to the architecture of the pre-trained model.  

We evaluate our approach on four types of language understanding tasks – natural language inference, question answering, semantic similarity, and text classiﬁcation. Our general task-agnostic model outperforms disc rim i natively trained models that employ architectures speciﬁcally crafted for each task, signiﬁcantly improving upon the state of the art in 9 out of the 12 tasks studied. For instance, we achieve absolute improvements of  $8.9\%$   on commonsense reasoning (Stories Cloze Test) [ 40 ],  $5.7\%$   on question answering (RACE) [ 30 ],   $1.5\%$   on textual entailment (MultiNLI) [ 66 ] and   $5.5\%$   on the recently introduced GLUE multi-task benchmark [ 64 ]. We also analyzed zero-shot behaviors of the pre-trained model on four different settings and demonstrate that it acquires useful linguistic knowledge for downstream tasks.  

# 2 Related Work  

Semi-supervised learning for NLP Our work broadly falls under the category of semi-supervised learning for natural language. This paradigm has attracted signiﬁcant interest, with applications to tasks like sequence labeling [ 24 ,  33 ,  57 ] or text classiﬁcation [ 41 ,  70 ]. The earliest approaches used unlabeled data to compute word-level or phrase-level statistics, which were then used as features in a supervised model [33]. Over the last few years, researchers have demonstrated the beneﬁts of using word embeddings [ 11 ,  39 ,  42 ], which are trained on unlabeled corpora, to improve performance on a variety of tasks [ 8 ,  11 ,  26 ,  45 ]. These approaches, however, mainly transfer word-level information, whereas we aim to capture higher-level semantics.  

Recent approaches have investigated learning and utilizing more than word-level semantics from unlabeled data. Phrase-level or sentence-level embeddings, which can be trained using an unlabeled corpus, have been used to encode text into suitable vector representations for various target tasks [ 28 , 32, 1, 36, 22, 12, 56, 31].  

Unsupervised pre-training Unsupervised pre-training is a special case of semi-supervised learning where the goal is to ﬁnd a good initialization point instead of modifying the supervised learning objective. Early works explored the use of the technique in image classiﬁcation [ 20 ,  49 ,  63 ] and regression tasks [ 3 ]. Subsequent research [ 15 ] demonstrated that pre-training acts as a regularization scheme, enabling better generalization in deep neural networks. In recent work, the method has been used to help train deep neural networks on various tasks like image classiﬁcation [ 69 ], speech recognition [68], entity disambiguation [17] and machine translation [48].  

The closest line of work to ours involves pre-training a neural network using a language modeling objective and then ﬁne-tuning it on a target task with supervision. Dai et al. [ 13 ] and Howard and Ruder [ 21 ] follow this method to improve text classiﬁcation. However, although the pre-training phase helps capture some linguistic information, their usage of LSTM models restricts their prediction ability to a short range. In contrast, our choice of transformer networks allows us to capture longer- range linguistic structure, as demonstrated in our experiments. Further, we also demonstrate the effectiveness of our model on a wider range of tasks including natural language inference, paraphrase detection and story completion. Other approaches [ 43 ,  44 ,  38 ] use hidden representations from a pre-trained language or machine translation model as auxiliary features while training a supervised model on the target task. This involves a substantial amount of new parameters for each separate target task, whereas we require minimal changes to our model architecture during transfer.  

Auxiliary training objectives Adding auxiliary unsupervised training objectives is an alternative form of semi-supervised learning. Early work by Collobert and Weston [ 10 ] used a wide variety of auxiliary NLP tasks such as POS tagging, chunking, named entity recognition, and language modeling to improve semantic role labeling. More recently, Rei [ 50 ] added an auxiliary language modeling objective to their target task objective and demonstrated performance gains on sequence labeling tasks. Our experiments also use an auxiliary objective, but as we show, unsupervised pre-training already learns several linguistic aspects relevant to target tasks.  

# 3 Framework  

Our training procedure consists of two stages. The ﬁrst stage is learning a high-capacity language model on a large corpus of text. This is followed by a ﬁne-tuning stage, where we adapt the model to a discriminative task with labeled data.  

# 3.1 Unsupervised pre-training  

Given an unsupervised corpus of tokens    ${\mathcal{U}}=\{u_{1},.\,.\,.\,,u_{n}\}$  , we use a standard language modeling objective to maximize the following likelihood:  

$$
L_{1}(\mathcal{U})=\sum_{i}\log P(u_{i}|u_{i-k},.\,.\,,u_{i-1};\Theta)
$$  

where  $k$   is the size of the context window, and the conditional probability    $P$   is modeled using a neural network with parameters  $\Theta$  . These parameters are trained using stochastic gradient descent [51].  

In our experiments, we use a multi-layer  Transformer decoder  [ 34 ] for the language model, which is a variant of the transformer [ 62 ]. This model applies a multi-headed self-attention operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens:  

$$
\begin{array}{r l}&{\quad h_{0}=U W_{e}+W_{p}}\\ &{\quad h_{l}=\mathtt{t r a n s f o r m e r\_b l o c k}(h_{l-1})\forall i\in[1,n]}\\ &{\quad P(u)=\mathtt{s o f t m a x}(h_{n}W_{e}^{T})}\end{array}
$$  

where    $U=(u_{-k},.\,.\,.\,,u_{-1})$   is the context vector of tokens,  $n$   is the number of layers,  $W_{e}$   is the token embedding matrix, and  $W_{p}$   is the position embedding matrix.  

# 3.2 Supervised ﬁne-tuning  

After training the model with the objective in Eq. 1, we adapt the parameters to the supervised target ssume a labeled dataset  $\mathcal{C}$  , where each instance consists of a sequence of input tokens,  $x^{1},\cdot\cdot\cdot,x^{m}$  , along with a label    $y$  . The inputs are passed through our pre-trained model to obtain the ﬁnal transformer block’s activation  $h_{l}^{m}$    , which is then fed into an added linear output layer with parameters    $W_{y}$   to predict  $y$  :  

$$
P(y|x^{1},\cdot\,.\,.\,,x^{m})=\mathtt{s o f t m a x}(h_{l}^{m}W_{y}).
$$  

This gives us the following objective to maximize:  

$$
L_{2}(\mathcal C)=\sum_{(x,y)}\log P(y|x^{1},.\,.\,.\,,x^{m}).
$$  

We additionally found that including language modeling as an auxiliary objective to the ﬁne-tuning helped learning by (a) improving generalization of the supervised model, and (b) accelerating convergence. This is in line with prior work [ 50 ,  43 ], who also observed improved performance with such an auxiliary objective. Speciﬁcally, we optimize the following objective (with weight  $\lambda$  ):  

$$
L_{3}(\mathcal{C})=L_{2}(\mathcal{C})+\lambda*L_{1}(\mathcal{C})
$$  

Overall, the only extra parameters we require during ﬁne-tuning are  $W_{y}$  , and embeddings for delimiter tokens (described below in Section 3.3).  

![](images/c095c1d0d3782a6b24091a8f182c5f1cd7a251c861b0bb09b634a4336f7e42b3.jpg)  
Figure 1:  (left)  Transformer architecture and training objectives used in this work.  (right)  Input transformations for ﬁne-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.  

# 3.3Task-speciﬁc input transformations  

For some tasks, like text classiﬁcation, we can directly ﬁne-tune our model as described above. Certain other tasks, like question answering or textual entailment, have structured inputs such as ordered sentence pairs, or triplets of document, question, and answers. Since our pre-trained model was trained on contiguous sequences of text, we require some modiﬁcations to apply it to these tasks. Previous work proposed learning task speciﬁc architectures on top of transferred representations [ 44 ]. Such an approach re-introduces a signiﬁcant amount of task-speciﬁc customization and does not use transfer learning for these additional architectural components. Instead, we use a traversal-style approach [ 52 ], where we convert structured inputs into an ordered sequence that our pre-trained model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. We provide a brief description of these input transformations below and Figure 1 provides a visual illustration. All transformations include adding randomly initialized start and end tokens   $(\langle s\rangle,\langle e\rangle)$  .  

Textual entailment For entailment tasks, we concatenate the premise  $p$   and hypothesis  $h$   token sequences, with a delimiter token   $(\S)$   in between.  

Similarity For similarity tasks, there is no inherent ordering of the two sentences being compared. To reﬂect this, we modify the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each independently to produce two sequence representations  $h_{l}^{m}$  which are added element-wise before being fed into the linear output layer.  

Question Answering and Commonsense Reasoning For these tasks, we are given a context document  $z$  , a question  $q$  , and a set of possible answers    $\left\{a_{k}\right\}$  . We concatenate the  ontext and question with each possible answer, adding a delimiter token in between to get  $[z;q;\S;a_{k}]$  . Each of these sequences are processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.  

# 4 Experiments  

4.1Setup  

Unsupervised pre-training We use the BooksCorpus dataset [ 71 ] for training the language model. It contains over 7,000 unique unpublished books from a variety of genres including Adventure, Fantasy, and Romance. Crucially, it contains long stretches of contiguous text, which allows the generative model to learn to condition on long-range information. An alternative dataset, the 1B Word Benchmark, which is used by a similar approach, ELMo [ 44 ], is approximately the same size  

![Table 1: A list of the different tasks and datasets used in our experiments. ](images/aa0e8bcea4f0a8f65411c8a33b1e5f98d9322e21de86d4ca7c704dd7c7912210.jpg)  

but is shufﬂed at a sentence level - destroying long-range structure. Our language model achieves a very low token level perplexity of 18.4 on this corpus.  

Model speciﬁcations Our model largely follows the original transformer work [ 62 ]. We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states. We used the Adam optimization scheme [ 27 ] with a max learning rate of  $2.5\mathrm{e}–4$  . The learning rate was increased linearly from zero over the ﬁrst 2000 updates and annealed to 0 using a cosine schedule. We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens. Since layernorm [ 2 ] is used extensively throughout the model, a simple weight initialization of  $N(0,0.{\overset{.}{02}})$   was sufﬁcient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges [ 53 ] and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. We also employed a modiﬁed version of L2 regularization proposed in [ 37 ], with    $w=0.01$   on all non bias or gain weights. For the activation function, we used the Gaussian Error Linear Unit (GELU) [ 18 ]. We used learned position embeddings instead of the sinusoidal version proposed in the original work. We use the  ftfy  library 2   to clean the raw text in BooksCorpus, standardize some punctuation and whitespace, and use the  spaCy  tokenizer.  

Fine-tuning details Unless speciﬁed, we reuse the hyperparameter settings from unsupervised pre-training. We add dropout to the classiﬁer with a rate of 0.1. For most tasks, we use a learning rate of  $6.25\mathrm{e}–5$   and a batchsize of 32. Our model ﬁnetunes quickly and 3 epochs of training was sufﬁcient for most cases. We use a linear learning rate decay schedule with warmup over   $0.2\%$   of training.    $\lambda$  was set to 0.5.  

# 4.2 Supervised ﬁne-tuning  

We perform experiments on a variety of supervised tasks including natural language inference, question answering, semantic similarity, and text classiﬁcation. Some of these tasks are available as part of the recently released GLUE multi-task benchmark [ 64 ], which we make use of. Figure 1 provides an overview of all the tasks and datasets.  

Natural Language Inference The task of natural language inference (NLI), also known as recog- nizing textual entailment, involves reading a pair of sentences and judging the relationship between them from one of  entailment ,  contradiction  or  neutral . Although there has been a lot of recent interest [ 58 ,  35 ,  44 ], the task remains challenging due to the presence of a wide variety of phenomena like lexical entailment, coreference, and lexical and syntactic ambiguity. We evaluate on ﬁve datasets with diverse sources, including image captions (SNLI), transcribed speech, popular ﬁction, and government reports (MNLI), Wikipedia articles (QNLI), science exams (SciTail) or news articles (RTE).  

Table 2 details various results on the different NLI tasks for our model and previous state-of-the-art approaches. Our method signiﬁcantly outperforms the baselines on four of the ﬁve datasets, achieving absolute improvements of upto   $1.5\%$   on MNLI,   $5\%$   on SciTail,   $5.8\%$   on QNLI and   $0.6\%$   on SNLI over the previous best results. This demonstrates our model’s ability to better reason over multiple sentences, and handle aspects of linguistic ambiguity. On RTE, one of the smaller datasets we evaluate on (2490 examples), we achieve an accuracy of  $56\%$  , which is below the  $61.7\%$   reported by a multi-task biLSTM model. Given the strong performance of our approach on larger NLI datasets, it is likely our model will beneﬁt from multi-task training as well but we have not explored this currently.  

![Table 3: Results on question answering and commonsense reasoning, comparing our model with current state-of-the-art methods..  $9\mathbf{x}$   means an ensemble of 9 models. ](images/fc9870c8e88772adda6acd710356a028cdb992de798da48ae8d62babe2bd352a.jpg)  

![](images/4ad342c91e123fe6988d40fe92f2ac64c1cabfe89628ca524c6ac409deb1f3df.jpg)  

Question answering and commonsense reasoning Another task that requires aspects of single and multi-sentence reasoning is question answering. We use the recently released RACE dataset [ 30 ], consisting of English passages with associated questions from middle and high school exams. This corpus has been shown to contain more reasoning type questions that other datasets like CNN [ 19 ] or SQuaD [ 47 ], providing the perfect evaluation for our model which is trained to handle long-range contexts. In addition, we evaluate on the Story Cloze Test [ 40 ], which involves selecting the correct ending to multi-sentence stories from two options. On these tasks, our model again outperforms the previous best results by signiﬁcant margins - up to   $8.9\%$   on Story Cloze, and  $5.7\%$   overall on RACE. This demonstrates the ability of our model to handle long-range contexts effectively.  

Semantic Similarity Semantic similarity (or paraphrase detection) tasks involve predicting whether two sentences are semantically equivalent or not. The challenges lie in recognizing rephrasing of concepts, understanding negation, and handling syntactic ambiguity. We use three datasets for this task – the Microsoft Paraphrase corpus (MRPC) [ 14 ] (collected from news sources), the Quora Question Pairs (QQP) dataset [ 9 ], and the Semantic Textual Similarity benchmark (STS-B) [ 6 ]. We obtain state-of-the-art results on two of the three semantic similarity tasks (Table 4) with a 1 point absolute gain on STS-B. The performance delta on QQP is signiﬁcant, with a  $4.2\%$   absolute improvement over Single-task BiLSTM  $\cdot+\operatorname{E}$  LMo  $^+$   Attn.  

Classiﬁcation Finally, we also evaluate on two different text classiﬁcation tasks. The Corpus of Linguistic Acceptability (CoLA) [ 65 ] contains expert judgements on whether a sentence is grammatical or not, and tests the innate linguistic bias of trained models. The Stanford Sentiment Treebank (SST-2) [ 54 ], on the other hand, is a standard binary classiﬁcation task. Our model obtains an score of 45.4 on CoLA, which is an especially big jump over the previous best result of 35.0, showcasing the innate linguistic bias learned by our model. The model also achieves   $91.3\%$   accuracy on SST-2, which is competitive with the state-of-the-art results. We also achieve an overall score of 72.8 on the GLUE benchmark, which is signiﬁcantly better than the previous best of 68.9.  

![Table 4: Semantic similarity and classiﬁcation results, comparing our model with current state-of-the- art methods. All task evaluations in this table were done using the GLUE benchmark. (  $m c=$   Mathews correlation,    $a c c{=}I$  Accuracy,  $p c{=}$  Pearson correlation) ](images/1b1354a9dcda24d620dcf4491f8a8bf2d110f067bba504a6568f3cb899a124b6.jpg)  

Overall, our approach achieves new state-of-the-art results in 9 out of the 12 datasets we evaluate on, outperforming ensembles in many cases. Our results also indicate that our approach works well across datasets of differen from smaller datasets such as STS-B   $({\approx}5.7\mathrm{k}$   training examples) – to the largest one – SNLI ( 550k training examples).  

# 5 Analysis  

Impact of number of layers transferred We observed the impact of transferring a variable number of layers from unsupervised pre-training to the supervised target task. Figure 2(left) illustrates the performance of our approach on MultiNLI and RACE as a function of the number of layers transferred. We observe the standard result that transferring embeddings improves performance and that each transformer layer provides further beneﬁts up to  $9\%$   for full transfer on MultiNLI. This indicates that each layer in the pre-trained model contains useful functionality for solving target tasks.  

![](images/8a5ac308ab8c8cb1f1c4cdd842d3d8c7c6d19ff5431ba6edbfd49007238cee72.jpg)  
Figure 2: ( left ) Effect of transferring increasing number of layers from the pre-trained language model on RACE and MultiNLI. ( right ) Plot showing the evolution of zero-shot performance on different tasks as a function of LM pre-training updates. Performance per task is normalized between a random guess baseline and the current state-of-the-art with a single model.  

![Table 5: Analysis of various model ablations on different tasks. Avg. score is a unweighted average of all the results. (  $m c{=}$   Mathews correlation,  acc =Accuracy,  $p c{=}$  Pearson correlation) ](images/9a2226121034b4aee5659d49b47ae1f048c11929a11078a3517cadcdee3941eb.jpg)  

attentional memory of the transformer assists in transfer compared to LSTMs. We designed a series of heuristic solutions that use the underlying generative model to perform tasks without supervised ﬁnetuning. We visualize the effectiveness of these heuristic solutions over the course of generative pre-training in Fig 2(right). We observe the performance of these heuristics is stable and steadily increases over training suggesting that generative pretraining supports the learning of a wide variety of task relevant functionality. We also observe the LSTM exhibits higher variance in its zero-shot performance suggesting that the inductive bias of the Transformer architecture assists in transfer.  

For CoLA (linguistic acceptability), examples are scored as the average token log-probability the generative model assigns and predictions are made by thresholding. For SST-2 (sentiment analysis), we append the token  very  to each example and restrict the language model’s output distribution to only the words  positive  and  negative  and guess the token it assigns higher probability to as the prediction. For RACE (question answering), we pick the answer the generative model assigns the highest average token log-probability when conditioned on the document and question. For DPRD [ 46 ] (winograd schemas), we replace the deﬁnite pronoun with the two possible referrents and predict the resolution that the generative model assigns higher average token log-probability to the rest of the sequence after the substitution.  

Ablation studies We perform three different ablation studies (Table 5). First, we examine the performance of our method without the auxiliary LM objective during ﬁne-tuning. We observe that the auxiliary objective helps on the NLI tasks and QQP. Overall, the trend suggests that larger datasets beneﬁt from the auxiliary objective but smaller datasets do not. Second, we analyze the effect of the Transformer by comparing it with a single layer 2048 unit LSTM using the same framework. We observe a 5.6 average score drop when using the LSTM instead of the Transformer. The LSTM only outperforms the Transformer on one dataset – MRPC. Finally, we also compare with our transformer architecture directly trained on supervised target tasks, without pre-training. We observe that the lack of pre-training hurts performance across all the tasks, resulting in a   $14.8\%$   decrease compared to our full model.  

# 6 Conclusion  

We introduced a framework for achieving strong natural language understanding with a single task-agnostic model through generative pre-training and discriminative ﬁne-tuning. By pre-training on a diverse corpus with long stretches of contiguous text our model acquires signiﬁcant world knowledge and ability to process long-range dependencies which are then successfully transferred to solving discriminative tasks such as question answering, semantic similarity assessment, entailment determination, and text classiﬁcation, improving the state of the art on 9 of the 12 datasets we study. Using unsupervised (pre-)training to boost performance on discriminative tasks has long been an important goal of Machine Learning research. Our work suggests that achieving signiﬁcant performance gains is indeed possible, and offers hints as to what models (Transformers) and data sets (text with long range dependencies) work best with this approach. We hope that this will help enable new research into unsupervised learning, for both natural language understanding and other domains, further improving our understanding of how and when unsupervised learning works.  

# References  

[1] S. Arora, Y. Liang, and T. Ma. A simple but tough-to-beat baseline for sentence embeddings. 2016.  

[2] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization.  arXiv preprint arXiv:1607.06450 , 2016.  

[3]  Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle. Greedy layer-wise training of deep networks. In Advances in neural information processing systems , pages 153–160, 2007.

 [4]  L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo. The ﬁfth pascal recognizing textual entailment challenge. In  TAC , 2009.

 [5]  S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning natural language inference.  EMNLP , 2015.

 [6]  D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation.  arXiv preprint arXiv:1708.00055 , 2017.

 [7]  S. Chaturvedi, H. Peng, and D. Roth. Story comprehension for predicting what happens next. In  Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing , pages 1603–1614, 2017.

 [8]  D. Chen and C. Manning. A fast and accurate dependency parser using neural networks. In  Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) , pages 740–750, 2014.

 [9]  Z. Chen, H. Zhang, X. Zhang, and L. Zhao. Quora question pairs. https://data.quora.com/First-Quora- Dataset-Release-Question-Pairs, 2018.

 [10]  R. Collobert and J. Weston. A uniﬁed architecture for natural language processing: Deep neural networks with multitask learning. In  Proceedings of the 25th international conference on Machine learning , pages 160–167. ACM, 2008.

 [11]  R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa. Natural language processing (almost) from scratch.  Journal of Machine Learning Research , 12(Aug):2493–2537, 2011.

 [12]  A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. Supervised learning of universal sentence representations from natural language inference data.  EMNLP , 2017.

 [13]  A. M. Dai and Q. V. Le. Semi-supervised sequence learning. In  Advances in Neural Information Processing Systems , pages 3079–3087, 2015.

 [14]  W. B. Dolan and C. Brockett. Automatically constructing a corpus of sentential paraphrases. In  Proceedings of the Third International Workshop on Paraphrasing (IWP2005) , 2005.

 [15]  D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. Why does unsupervised pre-training help deep learning?  Journal of Machine Learning Research , 11(Feb):625–660, 2010.

 [16] S. Gray, A. Radford, and K. P. Diederik. Gpu kernels for block-sparse weights. 2017.

 [17]  Z. He, S. Liu, M. Li, M. Zhou, L. Zhang, and H. Wang. Learning entity representation for entity disam- biguation. In  Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) , volume 2, pages 30–34, 2013.

 [18]  D. Hendrycks and K. Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units.  arXiv preprint arXiv:1606.08415 , 2016.

 [19]  K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom. Teaching machines to read and comprehend. In  Advances in Neural Information Processing Systems , pages 1693– 1701, 2015.

 [20]  G. E. Hinton, S. Osindero, and Y.-W. Teh. A fast learning algorithm for deep belief nets.  Neural computation , 18(7):1527–1554, 2006.

 [21]  J. Howard and S. Ruder. Universal language model ﬁne-tuning for text classiﬁcation.  Association for Computational Linguistics (ACL) , 2018.

 [22]  Y. Jernite, S. R. Bowman, and D. Sontag. Discourse-based objectives for fast unsupervised sentence representation learning.  arXiv preprint arXiv:1705.00557 , 2017.

 [23]  Y. Ji and J. Eisenstein. Discriminative improvements to distributional sentence similarity. In  Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing , pages 891–896, 2013.  

[24]  F. Jiao, S. Wang, C.-H. Lee, R. Greiner, and D. Schuurmans. Semi-supervised conditional random ﬁelds for improved sequence segmentation and labeling. In  Proceedings of the 21st International Conference on Computational Linguistics and the 44th annual meeting of the Association for Computational Linguistics , pages 209–216. Association for Computational Linguistics, 2006.

 [25]  T. Khot, A. Sabharwal, and P. Clark. Scitail: A textual entailment dataset from science question answering. In  Proceedings of AAAI , 2018.

 [26] Y. Kim. Convolutional neural networks for sentence classiﬁcation.  EMNLP , 2014.

 [27]  D. P. Kingma and J. Ba. Adam: A method for stochastic optimization.  arXiv preprint arXiv:1412.6980 , 2014.

 [28]  R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, and S. Fidler. Skip-thought vectors. In  Advances in neural information processing systems , pages 3294–3302, 2015.

 [29] N. Kitaev and D. Klein. Constituency parsing with a self-attentive encoder.  ACL , 2018.

 [30]  G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy. Race: Large-scale reading comprehension dataset from examinations.  EMNLP , 2017.

 [31]  G. Lample, L. Denoyer, and M. Ranzato. Unsupervised machine translation using monolingual corpora only.  ICLR , 2018.

 [32]  Q. Le and T. Mikolov. Distributed representations of sentences and documents. In  International Conference on Machine Learning , pages 1188–1196, 2014.

 [33]  P. Liang.  Semi-supervised learning for natural language . PhD thesis, Massachusetts Institute of Technology, 2005.

 [34]  P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. Generating wikipedia by summarizing long sequences.  ICLR , 2018.

 [35]  X. Liu, K. Duh, and J. Gao. Stochastic answer networks for natural language inference.  arXiv preprint arXiv:1804.07888 , 2018.

 [36] L. Logeswaran and H. Lee. An efﬁcient framework for learning sentence representations.  ICLR , 2018.

 [37]  I. Loshchilov and F. Hutter. Fixing weight decay regularization in adam.  arXiv preprint arXiv:1711.05101 , 2017.

 [38]  B. McCann, J. Bradbury, C. Xiong, and R. Socher. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems , pages 6297–6308, 2017.

 [39]  T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositional it y. In  Advances in neural information processing systems , pages 3111–3119, 2013.

 [40]  N. Mostafazadeh, M. Roth, A. Louis, N. Chambers, and J. Allen. Lsdsem 2017 shared task: The story cloze test. In  Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics , pages 46–51, 2017.

 [41]  K. Nigam, A. McCallum, and T. Mitchell. Semi-supervised text classiﬁcation using em.  Semi-Supervised Learning , pages 33–56, 2006.

 [42]  J. Pennington, R. Socher, and C. Manning. Glove: Global vectors for word representation. In  Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) , pages 1532–1543, 2014.

 [43] M. E. Peters, W. Ammar, C. Bhagavatula, and R. Power. Semi-supervised sequence tagging with bidirec- tional language models.  ACL , 2017.

 [44]  M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer. Deep contextual- ized word representations.  NAACL , 2018.

 [45]  Y. Qi, D. S. Sachan, M. Felix, S. J. Padmanabhan, and G. Neubig. When and why are pre-trained word embeddings useful for neural machine translation?  NAACL , 2018.  

[46]  A. Rahman and V. Ng. Resolving complex cases of deﬁnite pronouns: the winograd schema challenge. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning , pages 777–789. Association for Computational Linguistics, 2012.

 [47]  P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. Squad:   $100{,}000{+}$   questions for machine comprehension of text.  EMNLP , 2016.

 [48]  P. Ramachandran, P. J. Liu, and Q. V. Le. Unsupervised pretraining for sequence to sequence learning. arXiv preprint arXiv:1611.02683 , 2016.

 [49]  M. Ranzato, C. Poultney, S. Chopra, and Y. LeCun. Efﬁcient learning of sparse representations with an energy-based model. In  Advances in neural information processing systems , pages 1137–1144, 2007.

 [50] M. Rei. Semi-supervised multitask learning for sequence labeling.  ACL , 2017.

 [51]  H. Robbins and S. Monro. A stochastic approximation method.  The annals of mathematical statistics , pages 400–407, 1951.

 [52]  T. Rocktäschel, E. Grefenstette, K. M. Hermann, T. Koˇ cisk y, and P. Blunsom. Reasoning about entailment with neural attention.  arXiv preprint arXiv:1509.06664 , 2015.

 [53]  R. Sennrich, B. Haddow, and A. Birch. Neural machine translation of rare words with subword units.  arXiv preprint arXiv:1508.07909 , 2015.

 [54]  R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts. Recursive deep models for semantic compositional it y over a sentiment treebank. In  Proceedings of the 2013 conference on empirical methods in natural language processing , pages 1631–1642, 2013.

 [55]  S. Srinivasan, R. Arora, and M. Riedl. A simple and effective approach to the story cloze test.  arXiv preprint arXiv:1803.05547, 2018.

[56]  S. Subramanian, A. Trischler, Y. Bengio, and C. J. Pal. Learning general purpose distributed sentence representations via large scale multi-task learning.  arXiv preprint arXiv:1804.00079 , 2018.

 [57]  J. Suzuki and H. Isozaki. Semi-supervised sequential labeling and segmentation using giga-word scale unlabeled data.  Proceedings of ACL-08: HLT , pages 665–673, 2008.

 [58]  Y. Tay, L. A. Tuan, and S. C. Hui. A compare-propagate architecture with alignment factorization for natural language inference.  arXiv preprint arXiv:1801.00102 , 2017.

 [59]  Y. Tay, L. A. Tuan, and S. C. Hui. Multi-range reasoning for machine comprehension.  arXiv preprint arXiv:1803.09074 , 2018.

 [60]  J. Tian, Z. Zhou, M. Lan, and Y. Wu. Ecnu at semeval-2017 task 1: Leverage kernel-based traditional nlp features and neural networks to build a universal model for multilingual and cross-lingual semantic textual similarity. In  Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017) , pages 191–197, 2017.

 [61] Y. Tsvetkov. Opportunities and challenges in working with low-resource languages.  CMU , 2017.

 [62]  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In  Advances in Neural Information Processing Systems , pages 6000–6010, 2017.

 [63]  P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In  Proceedings of the 25th international conference on Machine learning , pages 1096–1103. ACM, 2008.

 [64]  A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding.  arXiv preprint arXiv:1804.07461 , 2018.

 [65]  A. Warstadt, A. Singh, and S. R. Bowman. Corpus of linguistic acceptability. http://nyu-mll.github.io/cola, 2018.

 [66]  A. Williams, N. Nangia, and S. R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference.  NAACL , 2018.

 [67]  Y. Xu, J. Liu, J. Gao, Y. Shen, and X. Liu. Towards human-level machine reading comprehension: Reasoning and inference with multiple strategies.  arXiv preprint arXiv:1711.04964 , 2017.  

[68]  D. Yu, L. Deng, and G. Dahl. Roles of pre-training and ﬁne-tuning in context-dependent dbn-hmms for real-world speech recognition. In  Proc. NIPS Workshop on Deep Learning and Unsupervised Feature Learning , 2010.

 [69]  R. Zhang, P. Isola, and A. A. Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In  CVPR , volume 1, page 6, 2017.

 [70] X. Zhu. Semi-supervised learning literature survey. 2005.

 [71]  Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, and S. Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In  Proceedings of the IEEE international conference on computer vision , pages 19–27, 2015.  