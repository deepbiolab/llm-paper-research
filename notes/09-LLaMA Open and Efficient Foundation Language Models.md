# LLaMA: Open and Efﬁcient Foundation Language Models  

# Hugo Touvron , Thibaut Lavril , Gautier Izacard , Xavier Martinet Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin Edouard Grave , Guillaume Lample ∗  

Meta AI  

# Abstract  

We introduce LLaMA, a collection of founda- tion language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly avail- able datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA- 65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community 1 .  

# 1 Introduction  

Large Languages Models (LLMs) trained on mas- sive corpora of texts have shown their ability to per- form new tasks from textual instructions or from a few examples ( Brown et al. ,  2020 ). These few-shot properties ﬁrst appeared when scaling models to a sufﬁcient size ( Kaplan et al. ,  2020 ), resulting in a line of work that focuses on further scaling these models ( Chowdhery et al. ,  2022 ;  Rae et al. ,  2021 ). These efforts are based on the assumption that more parameters will lead to better performance. However, recent work from  Hoffmann et al.  ( 2022 ) shows that, for a given compute budget, the best performances are not achieved by the largest mod- els, but by smaller models trained on more data.  

The objective of the scaling laws from  Hoff- mann et al.  ( 2022 ) is to determine how to best scale the dataset and model sizes for a particular training  compute budget. However, this objective disregards the  inference  budget, which becomes critical when serving a language model at scale. In this context, given a target level of performance, the preferred model is not the fastest to train but the fastest at inference, and although it may be cheaper to train a large model to reach a certain level of performance, a smaller one trained longer will ultimately be cheaper at inference. For instance, although  Hoffmann et al.  ( 2022 ) recommends training a 10B model on 200B tokens, we ﬁnd that the performance of a 7B model continues to improve even after 1T tokens.  

The focus of this work is to train a series of language models that achieve the best possible per- formance at various inference budgets, by training on more tokens than what is typically used. The resulting models, called  LLaMA , ranges from 7B to 65B parameters with competitive performance compared to the best existing LLMs. For instance, LLaMA-13B outperforms GPT-3 on most bench- marks, despite being   $10\times$   smaller. We believe that this model will help democratize the access and study of LLMs, since it can be run on a single GPU. At the higher-end of the scale, our 65B-parameter model is also competitive with the best large lan- guage models such as Chinchilla or PaLM-540B.  

Unlike Chinchilla, PaLM, or GPT-3, we only use publicly available data, making our work com- patible with open-sourcing, while most existing models rely on data which is either not publicly available or undocumented (e.g. “Books – 2TB” or “Social media conversations”). There exist some exceptions, notably OPT ( Zhang et al. ,  2022 ), GPT-NeoX ( Black et al. ,  2022 ), BLOOM ( Scao et al. ,  2022 ) and GLM ( Zeng et al. ,  2022 ), but none that are competitive with PaLM-62B or Chinchilla.  

In the rest of this paper, we present an overview of the modiﬁcations we made to the transformer architecture ( Vaswani et al. ,  2017 ), as well as our training method. We then report the performance of our models and compare with others LLMs on a set of standard benchmarks. Finally, we expose some of the biases and toxicity encoded in our models, using some of the most recent benchmarks from the responsible AI community.  

Our training approach is similar to the methods described in previous work ( Brown et al. ,  2020 ; Chowdhery et al. ,  2022 ), and is inspired by the Chinchilla scaling laws ( Hoffmann et al. ,  2022 ). We train large transformers on a large quantity of textual data using a standard optimizer.  

# 2.1 Pre-training Data  

Our training dataset is a mixture of several sources, reported in Table  1 , that cover a diverse set of do- mains. For the most part, we reuse data sources that have been leveraged to train other LLMs, with the restriction of only using data that is publicly available, and compatible with open sourcing. This leads to the following mixture of data and the per- centage they represent in the training set:  

English CommonCrawl   $[67\%]$  . We preprocess ﬁve CommonCrawl dumps, ranging from 2017 to 2020, with the CCNet pipeline ( Wenzek et al. , 2020 ). This process deduplicates the data at the line level, performs language identiﬁcation with a fastText linear classiﬁer to remove non-English pages and ﬁlters low quality content with an n- gram language model. In addition, we trained a linear model to classify pages used as references in Wikipedia  v.s.  randomly sampled pages, and discarded pages not classiﬁed as references.  

$\mathbf{C4}\,[15\%]$  . During exploratory experiments, we observed that using diverse pre-processed Com- monCrawl datasets improves performance. We thus included the publicly available C4 dataset ( Raffel et al. ,  2020 ) in our data. The preprocessing of C4 also contains deduplication and language identiﬁ- cation steps: the main difference with CCNet is the quality ﬁltering, which mostly relies on heuris- tics such as presence of punctuation marks or the number of words and sentences in a webpage.  

Github   $[{\bf4.5\%}]$  . We use the public GitHub dataset available on Google BigQuery. We only kept projects that are distributed under the Apache, BSD and MIT licenses. Additionally, we ﬁltered low quality ﬁles with heuristics based on the line length or proportion of alphanumeric characters, and removed boilerplate, such as headers, with reg- ular expressions. Finally, we deduplicate the result- ing dataset at the ﬁle level, with exact matches.  

Wikipedia   $[{\bf4.5\,\%}]$  . We add Wikipedia dumps from the June-August 2022 period, covering 20  

![](images/088556d6cc404db671be51638a925583addfd548e586a0dc5d694049e9f31bb9.jpg)  
Table 1:  Pre-training data.  Data mixtures used for pre- training, for each subset we list the sampling propor- tion, number of epochs performed on the subset when training on  $1.4\mathrm{T}$   tokens, and disk size. The pre-training runs on 1T tokens have the same sampling proportion.  

languages, which use either the Latin or Cyrillic scripts:  bg ,  ca ,  cs ,  da ,  de ,  en ,  es ,  fr ,  hr ,  hu ,  it , nl ,  pl ,  pt ,  ro ,  ru ,  sl ,  sr ,  sv ,  uk . We process the data to remove hyperlinks, comments and other formatting boilerplate.  

Gutenberg and Books3   $[4.5\%]$  . We include two book corpora in our training dataset: the Guten- berg Project, which contains books that are in the public domain, and the Books3 section of TheP- ile ( Gao et al. ,  2020 ), a publicly available dataset for training large language models. We perform deduplication at the book level, removing books with more than  $90\%$   content overlap.  

ArXiv   $[{\pmb2}.{\pmb5}\,\%]$  . We process arXiv Latex ﬁles to add scientiﬁc data to our dataset. Following Lewkowycz et al.  ( 2022 ), we removed everything before the ﬁrst section, as well as the bibliography. We also removed the comments from the .tex ﬁles, and inline-expanded deﬁnitions and macros written by users to increase consistency across papers.  

Stack Exchange   $[2\%]$  . We include a dump of Stack Exchange, a website of high quality ques- tions and answers that covers a diverse set of do- mains, ranging from computer science to chemistry. We kept the data from the 28 largest websites, re- moved the HTML tags from text and sorted the answers by score (from highest to lowest).  

Tokenizer. We tokenize the data with the byte- pair encoding (BPE) algorithm ( Sennrich et al. , 2015 ), using the implementation from Sentence- Piece ( Kudo and Richardson ,  2018 ). Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters.  

![Table 2:  Model sizes, architectures, and optimization hyper-parameters. ](images/bfd20e93694bed13207eac4e371d14688fe5977b7c6c8932d3098ccf26951860.jpg)  

Overall, our entire training dataset contains roughly 1.4T tokens after tokenization. For most of our training data, each token is used only once dur- ing training, with the exception of the Wikipedia and Books domains, over which we perform ap- proximately two epochs.  

# 2.2 Architecture  

Following recent work on large language models, our network is based on the transformer architec- ture ( Vaswani et al. ,  2017 ). We leverage various improvements that were subsequently proposed, and used in different models such as PaLM. Here are the main difference with the original architec- ture, and where we were found the inspiration for this change (in bracket):  

Pre-normalization [GPT3]. To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing func- tion, introduced by  Zhang and Sennrich  ( 2019 ).  

SwiGLU activation function [PaLM]. We re- place the ReLU non-linearity by the SwiGLU ac- tivation function, introduced by  Shazeer  ( 2020 ) to improve the performance. We use a dimension of  $\textstyle{\frac{2}{3}}4d$   instead of  $4d$   as in PaLM.  

Rotary Embeddings [GPTNeo].  We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by  Su et al.  ( 2021 ), at each layer of the network.  

The details of the hyper-parameters for our dif- ferent models are given in Table  2 .  

# 2.3 Optimizer  

Our models are trained using the AdamW opti- mizer ( Loshchilov and Hutter ,  2017 ), with the fol- lowing hyper-parameters:    $\beta_{1}\,=\,0.9,\beta_{2}\,=\,0.95$  . We use a cosine learning rate schedule, such that the ﬁnal learning rate is equal to   $10\%$   of the maxi- mal learning rate. We use a weight decay of  0 . 1  and gradient clipping of  1 . 0 . We use  2 ,  000  warmup  

![](images/633e7a3b67b98ff483885ee9a3312b28e4e218eb3581c55990e2f1b2a36a451f.jpg)  
Figure 1:  Training loss over train tokens for the 7B, 13B, 33B, and 65 models.  LLaMA-33B and LLaMA- 65B were trained on   $1.4\mathrm{T}$   tokens. The smaller models were trained on   $1.0\mathrm{T}$   tokens. All models are trained with a batch size of 4M tokens.  

steps, and vary the learning rate and batch size with the size of the model (see Table  2  for details).  

# 2.4 Efﬁcient implementation  

We make several optimizations to improve the train- ing speed of our models. First, we use an efﬁcient implementation of the causal multi-head attention to reduce memory usage and runtime. This imple- mentation, available in the  xformers  library,   is inspired by  Rabe and Staats  ( 2021 ) and uses the backward from  Dao et al.  ( 2022 ). This is achieved by not storing the attention weights and not com- puting the key/query scores that are masked due to the causal nature of the language modeling task.  

To further improve training efﬁciency, we re- duced the amount of activations that are recom- puted during the backward pass with checkpoint- ing. More precisely, we save the activations that are expensive to compute, such as the outputs of linear layers. This is achieved by manually imple- menting the backward function for the transformer layers, instead of relying on the PyTorch autograd. To fully beneﬁt from this optimization, we need to  

![Table 3:  Zero-shot performance on Common Sense Reasoning tasks. ](images/6f3897685004ad28880c57be9fca8a289d3bc55275728dc43083cb10a5072884.jpg)  

reduce the memory usage of the model by using model and sequence parallelism, as described by Korthikanti et al.  ( 2022 ). Moreover, we also over- lap the computation of activations and the commu- nication between GPUs over the network (due to all_reduce  operations) as much as possible.  

When training a 65B-parameter model, our code processes around 380 tokens/sec/GPU on 2048 A100 GPU with 80GB of RAM. This means that training over our dataset containing 1.4T tokens takes approximately 21 days.  

# 3 Main results  

Following previous work ( Brown et al. ,  2020 ), we consider zero-shot and few-shot tasks, and report results on a total of 20 benchmarks:  

•  Zero-shot.  We provide a textual description of the task and a test example. The model either provides an answer using open-ended generation, or ranks the proposed answers. •  Few-shot.  We provide a few examples of the task (between 1 and 64) and a test example. The model takes this text as input and gener- ates the answer or ranks different options.  

We compare LLaMA with other foundation mod- els, namely the non-publicly available language models GPT-3 ( Brown et al. ,  2020 ), Gopher ( Rae et al. ,  2021 ), Chinchilla ( Hoffmann et al. ,  2022 ) and PaLM ( Chowdhery et al. ,  2022 ), as well as the open-sourced OPT models ( Zhang et al. ,  2022 ), GPT-J ( Wang and Komatsuzaki ,  2021 ), and GPT- Neo ( Black et al. ,  2022 ). In Section  4 , we also brieﬂy compare LLaMA with instruction-tuned models such as OPT-IML ( Iyer et al. ,  2022 ) and Flan-PaLM ( Chung et al. ,  2022 ).  

We evaluate LLaMA on free-form generation tasks and multiple choice tasks. In the multiple choice tasks, the objective is to select the most appropriate completion among a set of given op- tions, based on a provided context. We select the completion with the highest likelihood given the provided context. We follow  Gao et al.  ( 2021 ) and use the likelihood normalized by the number of characters in the completion, except for certain datasets (OpenBookQA, BoolQ), for which we fol- low  Brown et al.  ( 2020 ), and select a completion based on the likelihood normalized by the likeli- hood of the completion given “Answer:” as context: . P ( completion | context ) /P ( completion | “ Answer :”)  

![](images/71c1f5c832e499d0a7f7c75b0c39c4658fbe2891604f7991d8a99ef2ac2d6e58.jpg)  
Table 4:  Natural Questions.  Exact match performance.  

# 3.1 Common Sense Reasoning  

We consider eight standard common sense rea- soning benchmarks: BoolQ ( Clark et al. ,  2019 ), PIQA ( Bisk et al. ,  2020 ), SIQA ( Sap et al. ,  2019 ), HellaSwag ( Zellers et al. ,  2019 ), WinoGrande ( Sak- aguchi et al. ,  2021 ), ARC easy and challenge ( Clark et al. ,  2018 ) and OpenBookQA ( Mihaylov et al. , 2018 ). These datasets include Cloze and Winograd style tasks, as well as multiple choice question an- swering. We evaluate in the zero-shot setting as done in the language modeling community.  

In Table  3 , we compare with existing models of various sizes and report numbers from the cor- responding papers. First, LLaMA-65B outper- forms Chinchilla-70B on all reported benchmarks but BoolQ. Similarly, this model surpasses PaLM- 540B everywhere but on BoolQ and WinoGrande. LLaMA-13B model also outperforms GPT-3 on most benchmarks despite being   $10\times$   smaller.  

# 3.2 Closed-book Question Answering  

We compare LLaMA to existing large language models on two closed-book question answering benchmarks: Natural Questions ( Kwiatkowski et al. ,  2019 ) and TriviaQA ( Joshi et al. ,  2017 ). For both benchmarks, we report exact match perfor- mance in a closed book setting, i.e., where the mod- els do not have access to documents that contain evidence to answer the question. In Table  4 , we report performance on Natural Questions, and in Ta- ble  5 , we report on TriviaQA. On both benchmarks, LLaMA-65B achieve state-of-the-arts performance in the zero-shot and few-shot settings. More im- portantly, the LLaMA-13B is also competitive on these benchmarks with GPT-3 and Chinchilla, de- spite being  $5.10\times$   smaller. This model runs on a single V100 GPU during inference.  

![](images/3950409fbc25aab2da6f60ad4d8a73a1486165eeb951d1c272245eafc1475198.jpg)  
Table 5: TriviaQA.  Zero-shot and few-shot exact match performance on the ﬁltered dev set.  

# 3.3 Reading Comprehension  

We evaluate our models on the RACE reading com- prehension benchmark ( Lai et al. ,  2017 ). This dataset was collected from English reading com- prehension exams designed for middle and high  

![Table 6:  Reading Comprehension.  Zero-shot accu- racy. ](images/dd93cf3088d07c22faee416c2bb292601ef31ebc5183b6ddb9aed95723f39f02.jpg)  

school Chinese students. We follow the evaluation setup from  Brown et al.  ( 2020 ) and report results in Table  6 . On these benchmarks, LLaMA-65B is competitive with PaLM-540B, and, LLaMA-13B outperforms GPT-3 by a few percents.  

# 3.4 Mathematical reasoning  

We evaluate our models on two mathematical rea- soning benchmarks: MATH ( Hendrycks et al. , 2021 ) and GSM8k ( Cobbe et al. ,  2021 ). MATH is a dataset of 12K middle school and high school mathematics problems written in LaTeX. GSM8k is a set of middle school mathematical problems. In Table  7 , we compare with PaLM and Min- erva ( Lewkowycz et al. ,  2022 ). Minerva is a series of PaLM models ﬁnetuned on 38.5B tokens ex- tracted from ArXiv and Math Web Pages, while neither PaLM or LLaMA are ﬁnetuned on mathe- matical data. The numbers for PaLM and Minerva are taken from  Lewkowycz et al.  ( 2022 ), and we compare with and without  maj1@k .  maj1@k  de- notes evaluations where we generate    $k$   samples for each problem and perform a majority voting ( Wang et al. ,  2022 ). On GSM8k, we observe that LLaMA- 65B outperforms Minerva-62B, although it has not been ﬁne-tuned on mathematical data.  

# 3.5 Code generation  

We evaluate the ability of our models to write code from a natural language description on two benchmarks: HumanEval ( Chen et al. ,  2021 ) and MBPP ( Austin et al. ,  2021 ). For both tasks, the model receives a description of the program in a few sentences, as well as a few input-output ex- amples. In HumanEval, it also receives a function signature, and the prompt is formatted as natural code with the textual description and tests in a  

![](images/6f0ee70e99ef1e76ca9dde3410b47d8e12e6aa248217f4d81d2f40c168a59dad.jpg)  

Table 7:  Model performance on quantitative reason- ing datasets.  For majority voting, we use the same setup as Minerva, with    $k\,=\,256$   samples for MATH and    $k=100$   for GSM8k (Minerva 540B uses    $k=64$  for MATH and and    $k=40$   for GSM8k). LLaMA-65B outperforms Minerva 62B on GSM8k, although it has not been ﬁne-tuned on mathematical data.  

docstring. The model needs to generate a Python program that ﬁts the description and satisﬁes the test cases. In Table  8 , we compare the pass  $@\,1$  scores of our models with existing language mod- els that have not been ﬁnetuned on code, namely PaLM and LaMDA ( Thoppilan et al. ,  2022 ). PaLM and LLaMA were trained on datasets that contain a similar number of code tokens.  

As show in Table  8 , for a similar number of parameters, LLaMA outperforms other gen- eral models such as LaMDA and PaLM, which are not trained or ﬁnetuned speciﬁcally for code. LLaMA with 13B parameters and more outper- forms LaMDA 137B on both HumanEval and MBPP. LLaMA 65B also outperforms PaLM 62B, even when it is trained longer. The pass  $@\,1$   results reported in this table were obtained by sampling with temperature 0.1. The pass  $@100$   and pass  $@80$  metrics were obtained with temperature 0.8. We use the same method as  Chen et al.  ( 2021 ) to obtain unbiased estimates of the pass  $@{\mathbf{k}}$  .  

It is possible to improve the performance on code by ﬁnetuning on code-speciﬁc tokens. For instance, PaLM-Coder ( Chowdhery et al. ,  2022 ) increases the pass  $@\,1$   score of PaLM on HumanEval from  $26.2\%$   for PaLM to   $36\%$  . Other models trained speciﬁcally for code also perform better than gen- eral models on these tasks ( Chen et al. ,  2021 ;  Ni- jkamp et al. ,  2022 ;  Fried et al. ,  2022 ). Finetuning on code tokens is beyond the scope of this paper.  

![](images/892cc07f9a90fdce6ac94393775b17af344ed17d51b6ff71ef77bde259e4c345.jpg)  

Table 8:  Model performance for code generation. We report the pass  $@$   score on HumanEval and MBPP. HumanEval generations are done in zero-shot and MBBP with 3-shot prompts similar to  Austin et al. ( 2021 ). The values marked with   ∗ are read from ﬁgures in  Chowdhery et al.  ( 2022 ).  

# 3.6 Massive Multitask Language Understanding  

The massive multitask language understanding benchmark, or MMLU, introduced by  Hendrycks et al.  ( 2020 ) consists of multiple choice questions covering various domains of knowledge, includ- ing humanities, STEM and social sciences. We evaluate our models in the 5-shot setting, using the examples provided by the benchmark, and report results in Table  9 . On this benchmark, we observe that the LLaMA-65B is behind both Chinchilla- 70B and PaLM-540B by a few percent in average, and across most domains. A potential explanation is that we have used a limited amount of books and academic papers in our pre-training data, i.e., ArXiv, Gutenberg and Books3, that sums up to only 177GB, while these models were trained on up to 2TB of books. This large quantity of books used by Gopher, Chinchilla and PaLM may also explain why Gopher outperforms GPT-3 on this benchmark, while it is comparable on other benchmarks.  

# 3.7 Evolution of performance during training  

During training, we tracked the performance of our models on a few question answering and common sense benchmarks, and report them in Figure  2 . On most benchmarks, the performance improves steadily, and correlates with the training perplexity of the model (see Figure  1 ). The exceptions are SIQA and WinoGrande. Most notably, on SIQA, we observe a lot of variance in performance,  

![Table 9:  Massive Multitask Language Understanding (MMLU).  Five-shot accuracy. ](images/43d0ea21a452f223e09db84eaa9b6288a2d2de5c4a01112e018a50c42fc00a53.jpg)  

that may indicate that this benchmark is not reliable. On WinoGrande, the performance does not correlate as well with training perplexity: the LLaMA-33B and LLaMA-65B have similar performance during the training.  

# 4 Instruction Finetuning  

In this section, we show that brieﬂy ﬁnetuning on instructions data rapidly leads to improvements on MMLU. Although the non-ﬁnetuned version of LLaMA-65B is already able to follow basic in- structions, we observe that a very small amount of ﬁnetuning improves the performance on MMLU, and further improves the ability of the model to follow instructions. Since this is not the focus of this paper, we only conducted a single experiment following the same protocol as  Chung et al.  ( 2022 ) to train an instruct model, LLaMA-I.  

![](images/6d5bfe3094cf94ffe01f7e5d17c824dbfe8e494282f0255487fa42452fc38334.jpg)  

Table 10:  Instruction ﬁnetuning – MMLU (5-shot). Comparison of models of moderate size with and with- out instruction ﬁnetuning on MMLU.  

In Table  10 , we report the results of our instruct model LLaMA-I on MMLU and compare with ex- isting instruction ﬁnetuned models of moderate sizes, namely, OPT-IML ( Iyer et al. ,  2022 ) and the Flan-PaLM series ( Chung et al. ,  2022 ). All the re- ported numbers are from the corresponding papers. Despite the simplicity of the instruction ﬁnetuning approach used here, we reach   $68.9\%$   on MMLU. LLaMA-I (65B) outperforms on MMLU existing instruction ﬁnetuned models of moderate sizes, but are still far from the state-of-the-art, that is 77.4 for GPT  code-davinci-002  on MMLU (numbers taken from  Iyer et al.  ( 2022 )). The details of the performance on MMLU on the 57 tasks can be found in Table  16  of the appendix.  

# 5 Bias, Toxicity and Misinformation  

Large language models have been showed to re- produce and amplify biases that are existing in the training data ( Sheng et al. ,  2019 ;  Kurita et al. , 2019 ), and to generate toxic or offensive con- tent ( Gehman et al. ,  2020 ). As our training dataset contains a large proportion of data from the Web, we believe that it is crucial to determine the po- tential for our models to generate such content. To understand the potential harm of LLaMA-65B, we evaluate on different benchmarks that measure toxic content production and stereotypes detection. While we have selected some of the standard bench- marks that are used by the language model com- munity to indicate some of the issues with these models, these evaluations are not sufﬁcient to fully understand the risks associated with these models.  

![](images/bf16a5a19001fd77936cc2f5266c09ab9dd57e998d5f4a0546c002415fea8684.jpg)  
Figure 2:  Evolution of performance on question answering and common sense reasoning during training.  

# 5.1 Real Toxicity Prompts  

Language models can generate toxic language, e.g., insults, hate speech or threats. There is a very large range of toxic content that a model can generate, making a thorough evaluation challenging. Several recent work ( Zhang et al. ,  2022 ;  Hoffmann et al. , 2022 ) have considered the Real Toxicity Prompts benchmark ( Gehman et al. ,  2020 ) as an indicator of how toxic is their model. Real Toxicity Prompts consists of about  $100\mathbf{k}$   prompts that the model must complete; then a toxicity score is automatically evaluated by making a request to PerspectiveAPI   3 . We do not have control over the pipeline used by the third-party PerspectiveAPI, making comparison with previous models difﬁcult.  

For each of the  $100\mathbf{k}$   prompts, we greedily gen- erate with our models, and measure their toxic- ity score. The score per prompt ranges from 0 (non-toxic) to 1 (toxic). In Table  11 , we report our averaged score on basic and respectful prompt cat- egories of Real Toxicity Prompts. These scores are “comparable” with what we observe in the litera- ture (e.g., 0.087 for Chinchilla) but the method- ologies differ between these work and ours (in terms of sampling strategy, number of prompts and time of API). We observe that toxicity increases  

![](images/f1e6a910c4773d0b4cdacd30cf31f2be81dd28271b9ca595a4af3593ad297330.jpg)  

Table 11:  Real Toxicity Prompts.  We run a greedy de- coder on the 100k prompts from this benchmark. The “respectful” versions are prompts starting with “Com- plete the following sentence in a polite, respectful, and unbiased manner:”, and “Basic” is without it. Scores were obtained using the PerplexityAPI, with higher score indicating more toxic generations.  

with the size of the model, especially for Respect- ful prompts. This was also observed in previous work ( Zhang et al. ,  2022 ), with the notable excep- tion of  Hoffmann et al.  ( 2022 ) where they do not see a difference between Chinchilla and Gopher, despite different sizes. This could be explained by the fact that the larger model, Gopher, has worse performance than Chinchilla, suggesting that the relation between toxicity and model size may only apply within a model family.  

![](images/2c8f9322b5b11b6e222e284fdbccc2599f659fb875213caabfeb2bbbe000ce69.jpg)  
Table 12:  CrowS-Pairs.  We compare the level of bi- ases contained in LLaMA-65B with OPT-175B and GPT3-175B. Higher score indicates higher bias.  

# 5.2 CrowS-Pairs  

We evaluate the biases in our model on the CrowS- Pairs ( Nangia et al. ,  2020 ). This dataset allows to measure biases in 9 categories: gender, religion, race/color, sexual orientation, age, nationality, dis- ability, physical appearance and socioeconomic sta- tus. Each example is composed of a stereotype and an anti-stereotype, we measure the model prefer- ence for the stereotypical sentence using the per- plexity of both sentences in a zero-shot setting. Higher scores thus indicate higher bias. We com- pare with GPT-3 and OPT-175B in Table  12 .  

LLaMA compares slightly favorably to both models on average. Our model is particularly bi- ased in the religion category   $(+10\%$   compared to OPT-175B), followed by age and gender. We ex- pect these biases to come from CommonCrawl de- spite multiple ﬁltering steps.  

# 5.3 WinoGender  

To further investigate the biases of our model on the gender category, we look at the WinoGender benchmark ( Rudinger et al. ,  2018 ), a co-reference resolution dataset. WinoGender is made of Wino- grad schema, and biases are evaluated by determin- ing if a model co-reference resolution performance is impacted by the gender of the pronoun.  

More precisely, each sentence has three men- tions: an “occupation”, a “participant”, and a “pronoun” where the pronoun is co-referencing either the occupation or participant. We prompt the model to determine the co-reference relation and measure if it does so correctly according to the context of the sentence. The goal is to reveal if societal biases associated with occupations have been captured by the model. For example, a sentence in the WinoGender dataset is “The nurse notiﬁed the patient that his shift would be ending in an hour.”, which is followed by  ‘His’ refers to . We then compare the perplexity of the continuations  the nurse  and  the patient  to per- form co-reference resolution with the model. We evaluate the performance when using 3 pronouns: “her/her/she”, “his/him/he” and “their/them/some- one” (the different choices corresponding to the grammatical function of the pronoun.  

In Table  13 , we report the co-reference scores for the three different pronouns contained in the dataset. We observe that our model is signiﬁcantly better at performing co-reference resolution for the “their/them/someone” pronouns than for the “her/her/she” and “his/him/he” pronouns. A simi- lar observation was made in previous work ( Rae et al. ,  2021 ;  Hoffmann et al. ,  2022 ), and is likely indicative of gender bias. Indeed, in the case of the “her/her/she” and “his/him/he” pronouns, the model is probably using the majority gender of the occu- pation to perform co-reference resolution, instead of using the evidence of the sentence.  

To further investigate this hypothesis, we look at the set of “gotcha” cases for the “her/her/she” and “his/him/he” pronouns in the WinoGender dataset. Theses cases correspond to sentences in which the pronoun does not match the majority gender of the occupation, and the occupation is the correct answer. In Table  13 , we observe that our model, LLaMA-65B, makes more errors on the gotcha examples, clearly showing that it capture societal biases related to gender and occupation. The drop of performance exists for “her/her/she” and “his/him/he” pronouns, which is indicative of biases regardless of gender.  

# 5.4 TruthfulQA  

TruthfulQA ( Lin et al. ,  2021 ) aims to measure the truthfulness of a model, i.e., its ability to identify when a claim is true.  Lin et al.  ( 2021 ) consider the deﬁnition of “true” in the sense of “literal truth about the real world”, and not claims that are only true in the context of a belief system or tradition. This benchmark can evaluate the risks of a model to generate misinformation or false claims. The questions are written in diverse style, cover 38 cat- egories and are designed to be adversarial.  

![Table 13:  WinoGender.  Co-reference resolution ac- curacy for the LLaMA models, for different pronouns (“her/her/she” and “his/him/he”). We observe that our models obtain better performance on “their/them/some- one’ pronouns than on “her/her/she” and “his/him/he’, which is likely indicative of biases. ](images/414ae3d5983cacf843a0f2e9b17f146f7722e80b4c4f24423c0c59ad83320a95.jpg)  

![](images/1a8f3ccdaed995f6fdf0d0a76365b3cb6b918c0ba448065c9b9b5e7e2ba2b554.jpg)  

Table 14:  TruthfulQA.  We report the fraction of truth- ful and truthful\*informative answers, as scored by spe- cially trained models via the OpenAI API. We follow the QA prompt style used in  Ouyang et al.  ( 2022 ), and report the performance of GPT-3 from the same paper.  

In Table  14 , we report the performance of our models on both questions to measure truthful mod- els and the intersection of truthful and informative. Compared to GPT-3, our model scores higher in both categories, but the rate of correct answers is still low, showing that our model is likely to hallu- cinate incorrect answers.  

# 6 Carbon footprint  

The training of our models have consumed a mas- sive quantity of energy, responsible for the emis- sion of carbon dioxide. We follow the recent liter- ature on the subject and breakdown both the total energy consumption and the resulting carbon foot- print in Table  15 . We follow a formula for  Wu et al. ( 2022 ) to estimate the Watt-hour, Wh, needed to train a model, as well as the tons of carbon emis- sions,   ${\bf t}{\bf C}{\bf O}_{2}{\bf e}{\bf q}$  . For the Wh, we use the formula:  

where we set the Power Usage Effectiveness (PUE) at  1 . 1 . The resulting carbon emission depends on the location of the data center used to train the net- work. For instance, BLOOM uses a grid that emits  $0.057\;\mathrm{kg\;CO_{2}e q/K W h}$   leading to   $27\ \mathrm{tCO_{2eq}}$   and OPT a grid that emits   $0.231\;\mathrm{kg\,CO_{2}e q/K W h}$  , lead- ing to   $82\,\mathrm{tCO_{2}e q}$  . In this study, we are interested in comparing the cost in carbon emission of training of these models if they were trained in the same data center. Hence, we do not take the location of data center in consideration, and use, instead, the US national average carbon intensity factor of  $0.385\;\mathrm{kg\,CO_{2}e q/K W h}$  . This leads to the following formula for the tons of carbon emissions:  

$$
\mathrm{tCO_{2}e q=M W h\times0.385.}
$$  

We apply the same formula to OPT and BLOOM for fair comparison. For OPT, we assume training required 34 days on 992 A100-80B (see their   $\mathrm{{logs^{4}}}$  ). Finally, we estimate that we used 2048 A100-80GB for a period of approximately 5 months to develop our models. This means that developing these mod- els would have cost around 2,638 MWh under our assumptions, and a total emission of   $^{1,015\,\mathrm{tCO_{2}e q}}$  . We hope that releasing these models will help to reduce future carbon emission since the training is already done, and some of the models are relatively small and can be run on a single GPU.  

# 7 Related work  

Language models are probability distributions over sequences of words, tokens or charac- ters ( Shannon ,  1948 ,  1951 ). This task, often framed as next token prediction, has long been considered a core problem in natural language processing ( Bahl et al. ,  1983 ;  Brown et al. ,  1990 ). Because  Turing ( 1950 ) proposed to measure machine intelligence by using language through the “imitation game”, language modeling has been proposed as a bench- mark to measure progress toward artiﬁcial intelli- gence ( Mahoney ,  1999 ).  

Architecture. Traditionally, language models were based on    $n$  -gram count statistics ( Bahl et al. ,  1983 ), and various smoothing techniques were proposed to improve the estimation of rare events ( Katz ,  1987 ;  Kneser and Ney ,  1995 ). In the past two decades, neural networks have been suc- cessfully applied to the language modelling task,  

![](images/255efb4c01e999e1bdd65fb8d1fc722041272ec903d2f0ae0a6a460c6bb96e40.jpg)  
Table 15:  Carbon footprint of training different models in the same data center.  We follow  Wu et al.  ( 2022 ) to compute carbon emission of training OPT, BLOOM and our models in the same data center. For the power consumption of a A100-80GB, we take the thermal design power for NVLink systems, that is 400W. We take a PUE of 1.1 and a carbon intensity factor set at the national US average of  $0.385\;\mathrm{kg}\;\mathrm{CO_{2}e}$   per KWh.  

starting from feed forward models ( Bengio et al. , 2000 ), recurrent neural networks ( Elman ,  1990 ; Mikolov et al. ,  2010 ) and LSTMs ( Hochreiter and Schmidhuber ,  1997 ;  Graves ,  2013 ). More recently, transformer networks, based on self-attention, have led to important improvements, especially for cap- turing long range dependencies ( Vaswani et al. , 2017 ;  Radford et al. ,  2018 ;  Dai et al. ,  2019 ).  

Scaling. There is a long history of scaling for language models, for both the model and dataset sizes.  Brants et al.  ( 2007 ) showed the beneﬁts of using language models trained on 2 trillion tokens, resulting in 300 billion    $n$  -grams, on the quality of machine translation. While this work relied on a simple smoothing technique, called  Stupid Backoff , Heaﬁeld et al.  ( 2013 ) later showed how to scale Kneser-Ney smoothing to Web-scale data. This allowed to train a 5-gram model on 975 billions to- kens from CommonCrawl, resulting in a model with 500 billions    $n$  -grams ( Buck et al. ,  2014 ). Chelba et al.  ( 2013 ) introduced the  One Billion Word  benchmark, a large scale training dataset to measure the progress of language models.  

In the context of neural language models,  Joze- fowicz et al.  ( 2016 ) obtained state-of-the-art re- sults on the Billion Word benchmark by scaling LSTMs to 1 billion parameters. Later, scaling transformers lead to improvement on many NLP tasks. Notable models include BERT ( Devlin et al. , 2018 ), GPT-2 ( Radford et al. ,  2019 ), Megatron- LM ( Shoeybi et al. ,  2019 ), and T5 ( Raffel et al. , 2020 ). A signiﬁcant breakthrough was obtained with GPT-3 ( Brown et al. ,  2020 ), a model with 175 billion parameters. This lead to a series of Large Language Models , such as Jurassic-1 ( Lieber et al. ,  2021 ), Megatron-Turing NLG ( Smith et al. , 2022 ), Gopher ( Rae et al. ,  2021 ), Chinchilla ( Hoff- mann et al. ,  2022 ), PaLM ( Chowdhery et al. ,  2022 ), OPT ( Zhang et al. ,  2022 ), and GLM ( Zeng et al. , 2022 ).  Hestness et al.  ( 2017 ) and  Rosenfeld et al. ( 2019 ) studied the impact of scaling on the perfor- mance of deep learning models, showing the exis- tence of power laws between the model and dataset sizes and the performance of the system.  Kaplan et al.  ( 2020 ) derived power laws speciﬁcally for transformer based language models, which were later reﬁned by  Hoffmann et al.  ( 2022 ), by adapting the learning rate schedule when scaling datasets. Finally,  Wei et al.  ( 2022 ) studied the effect of scal- ing on the abilities of large language models.  

# 8 Conclusion  

In this paper, we presented a series of language models that are released openly, and competitive with state-of-the-art foundation models. Most notably, LLaMA-13B outperforms GPT-3 while being more than   $10\times$   smaller, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. Unlike previous studies, we show that it is possible to achieve state-of-the-art performance by training exclusively on publicly available data, without resorting to proprietary datasets. We hope that releasing these models to the research community will accelerate the development of large language models, and help efforts to improve their robust- ness and mitigate known issues such as toxicity and bias. Additionally, we observed like  Chung et al. ( 2022 ) that ﬁnetuning these models on instructions lead to promising results, and we plan to further investigate this in future work. Finally, we plan to release larger models trained on larger pretraining corpora in the future, since we have seen a constant improvement in performance as we were scaling.  

# Acknowledgements  

We thank Daniel Haziza, Francisco Massa, Jeremy Reizenstein, Artem Korenev, and Patrick Labatut from the xformers team. We thank Susan Zhang and Stephen Roller for their support on data deduplication. We thank Luca Wehrstedt, Vegard Mella, and Pierre-Emmanuel Mazaré for their support on training stability. We thank Shubho Sengupta, Kalyan Saladi, and all the AI infra team for their support. We thank Jane Yu for her input on evaluation. We thank Yongyi Hu for his help on data collection.  

# References  

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models.  

Lalit R Bahl, Frederick Jelinek, and Robert L Mercer. 1983. A maximum likelihood approach to continu- ous speech recognition.  IEEE transactions on pat- tern analysis and machine intelligence , pages 179– 190.  

Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000. A neural probabilistic language model.  Ad- vances in neural information processing systems , 13.  

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physi- cal commonsense in natural language. In  Proceed- ings of the AAAI conference on artiﬁcial intelligence , pages 7432–7439.  

Sid Black, Stella Biderman, Eric Hallahan, Quentin An- thony, Leo Gao, Laurence Golding, Horace He, Con- nor Leahy, Kyle McDonell, Jason Phang, et al. 2022. Gpt-neox-20b: An open-source autoregressive lan- guage model.  arXiv preprint arXiv:2204.06745 .  

Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, and Jeffrey Dean. 2007.  Large language mod- els in machine translation . In  Proceedings of the 2007 Joint Conference on Empirical Methods in Nat- ural Language Processing and Computational Nat- ural Language Learning (EMNLP-CoNLL) , pages 858–867, Prague, Czech Republic. Association for Computational Linguistics.  

Peter F Brown, John Cocke, Stephen A Della Pietra, Vincent J Della Pietra, Frederick Jelinek, John Laf-ferty, Robert L Mercer, and Paul S Roossin. 1990. A statistical approach to machine translation.  Compu- tational linguistics , 16(2):79–85.  

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda  

Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam Mc- Candlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.  Language models are few-shot learn- ers .  

Christian Buck, Kenneth Heaﬁeld, and Bas Van Ooyen. 2014. N-gram counts and language models from the common crawl. In  LREC , volume 2, page 4.  

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robin- son. 2013. One billion word benchmark for measur- ing progress in statistical language modeling.  arXiv preprint arXiv:1312.3005 .  

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Ka- plan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sas- try, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cum- mings, Matthias Plappert, Fotios Chantzis, Eliza- beth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welin- der, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Eval- uating large language models trained on code.  

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vin- odkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghe- mawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fe- dus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankara- narayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Bren- nan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways .  

Hyung Won Chung, Le Hou, S. Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Al- bert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdh- ery, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Wei Yu, Vincent Zhao, Yanping Huang, An- drew M. Dai, Hongkun Yu, Slav Petrov, Ed Huai hsin Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc Le, and Jason Wei. 2022. Scal- ing instruction-ﬁnetuned language models. arXiv preprint arXiv:2210.11416 .  

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difﬁculty of natural yes/no questions.  arXiv preprint arXiv:1905.10044 .  

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question an- swering? try arc, the ai2 reasoning challenge.  arXiv preprint arXiv:1803.05457 .  

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training veriﬁers to solve math word problems.  arXiv preprint arXiv:2110.14168 .  

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Car- bonell, Quoc V Le, and Ruslan Salakhutdinov. 2019. Transformer-xl: Attentive language mod- els beyond a ﬁxed-length context. arXiv preprint arXiv:1901.02860 .  

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efﬁcient exact attention with io-awareness. arXiv preprint arXiv:2205.14135 .  

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understand- ing.  arXiv preprint arXiv:1810.04805 .  

Jeffrey L Elman. 1990. Finding structure in time.  Cog- nitive science , 14(2):179–211.  

Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen- tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022. Incoder: A generative model for code inﬁlling and synthesis.  arXiv preprint arXiv:2204.05999 .  

Leo Gao, Stella Biderman, Sid Black, Laurence Gold- ing, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The Pile: An  $800\mathrm{g}\flat$   dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027 .  

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPoﬁ, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff,  

Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. A framework for few-shot language model evalua- tion .  

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. 2020. Realtoxici- typrompts: Evaluating neural toxic degeneration in language models.  arXiv preprint arXiv:2009.11462 .  

Alex Graves. 2013. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850 .  

Kenneth Heaﬁeld, Ivan Pouzyrevsky, Jonathan H Clark, and Philipp Koehn. 2013. Scalable modiﬁed kneser- ney language model estimation. In  Proceedings of the 51st Annual Meeting of the Association for Com- putational Linguistics (Volume 2: Short Papers) , pages 690–696.  

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Stein- hardt. 2020. Measuring massive multitask language understanding.  arXiv preprint arXiv:2009.03300 .  

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 .  

Joel Hestness, Sharan Narang, Newsha Ardalani, Gre- gory Diamos, Heewoo Jun, Hassan Kianinejad, Md Patwary, Mostofa Ali, Yang Yang, and Yanqi Zhou. 2017. Deep learning scaling is predictable, empirically.  arXiv preprint arXiv:1712.00409 .  

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation , 9(8):1735–1780.  

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Si- monyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022.  Training compute-optimal large language models .  

Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Dániel Simig, Ping Yu, Kurt Shus- ter, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. 2022. Opt-iml: Scaling language model instruc- tion meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017 .  

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehen- sion.  arXiv preprint arXiv:1705.03551 .  

Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 2016. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410 . Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 . Slava Katz. 1987. Estimation of probabilities from sparse data for the language model component of a speech recognizer.  IEEE transactions on acoustics, speech, and signal processing , 35(3):400–401. Reinhard Kneser and Hermann Ney. 1995. Improved backing-off for m-gram language modeling. In  1995 international conference on acoustics, speech, and signal processing , volume 1, pages 181–184. IEEE. Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2022. Reducing ac- tivation recomputation in large transformer models. arXiv preprint arXiv:2205.05198 . Taku Kudo and John Richardson. 2018. Sentencepiece: A simple and language independent subword tok- enizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226 . Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan W Black, and Yulia Tsvetkov. 2019. Quantifying social bi- ases in contextual word representations. In  1st ACL Workshop on Gender Bias for Natural Language Processing . Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red- ﬁeld, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a bench- mark for question answering research.  Transactions of the Association for Computational Linguistics , 7:453–466. Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683 . Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022.  Solving quantitative reasoning prob- lems with language models . In  Advances in Neural Information Processing Systems . Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. 2021. Jurassic-1: Technical details and evaluation.  White Paper. AI21 Labs , 1. Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods.  arXiv preprint arXiv:2109.07958 .  

Ilya Loshchilov and Frank Hutter. 2017. Decou- pled weight decay regularization. arXiv preprint arXiv:1711.05101 . Matthew V Mahoney. 1999. Text compression as a test for artiﬁcial intelligence.  AAAI/IAAI , 970. Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct elec- tricity? a new dataset for open book question answer- ing.  arXiv preprint arXiv:1809.02789 . Tomas Mikolov, Martin Karaﬁát, Lukas Burget, Jan Cernock\` y, and Sanjeev Khudanpur. 2010. Recur- rent neural network based language model. In  In- terspeech , pages 1045–1048. Makuhari. Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-pairs: A chal- lenge dataset for measuring social biases in masked language models . In  EMNLP 2020 . Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large lan- guage model for code with multi-turn program syn- thesis.  arXiv preprint arXiv:2203.13474 . Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback . In  Advances in Neural Infor- mation Processing Systems . Markus N Rabe and Charles Staats. 2021. Self- attention does not need    $o(n^{2})$   memory. arXiv preprint arXiv:2112.05682 . Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language under- standing by generative pre-training. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Lan- guage models are unsupervised multitask learners. OpenAI blog , 1(8):9. Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susan- nah Young, Eliza Rutherford, Tom Hennigan, Ja- cob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Mari- beth Rauh, Po-Sen Huang, Amelia Glaese, Jo- hannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, An- tonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Ne- matzadeh, Elena Gribovskaya, Domenic Donato,  

Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cy- prien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hecht- man, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stan- way, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. 2021. Scal- ing language models: Methods, analysis & insights from training gopher .  

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a uniﬁed text-to-text trans- former.  The Journal of Machine Learning Research , 21(1):5485–5551.  

Jonathan S Rosenfeld, Amir Rosenfeld, Yonatan Be- linkov, and Nir Shavit. 2019. A constructive predic- tion of the generalization error across scales.  arXiv preprint arXiv:1909.12673 .  

Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. 2018. Gender bias in coreference resolution . In  NAACL-HLT 2018 .  

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat- ula, and Yejin Choi. 2021. Winogrande: An adver- sarial winograd schema challenge at scale.  Commu- nications of the ACM , 64(9):99–106.  

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. 2019. Socialiqa: Com- monsense reasoning about social interactions.  arXiv preprint arXiv:1904.09728 .  

Teven Le Scao, Angela Fan, Christopher Akiki, El- lie Pavlick, Suzana Ili´ c, Daniel Hesslow, Ro- man Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b- parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100 .  

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2015. Neural machine translation of rare words with subword units.  arXiv preprint arXiv:1508.07909 .  

Claude E Shannon. 1948. A mathematical theory of communication.  The Bell system technical journal , 27(3):379–423.  

Claude E Shannon. 1951. Prediction and entropy of printed english. Bell system technical journal , 30(1):50–64.  

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. 2019. The woman worked as a babysitter: On biases in language generation.  arXiv preprint arXiv:1909.01326 .  

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catan- zaro. 2019. Megatron-lm: Training multi-billion pa- rameter language models using model parallelism. arXiv preprint arXiv:1909.08053 .  

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. 2022. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model .  

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2021. Roformer: En- hanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864 .  

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, De- hao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinod- kumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman- John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. 2022. Lamda: Lan- guage models for dialog applications.  

A. M. Turing. 1950.  Computing Machinery and Intel- ligence . [Oxford University Press, Mind Associa- tion].  

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In  Advances in Neural Information Pro- cessing Systems 30 , pages 5998–6008.  

Ben Wang and Aran Komatsuzaki. 2021. GPT-J- 6B: A 6 Billion Parameter Autoregressive Lan- guage Model.  https://github.com/kingoflolz/ mesh-transformer-jax .  

Noam Shazeer. 2020. Glu variants improve trans- former.  arXiv preprint arXiv:2002.05202 .  

and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models.  

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682 .  

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Con- neau, Vishrav Chaudhary, Francisco Guzmán, Ar- mand Joulin, and Edouard Grave. 2020.  CCNet: Ex- tracting high quality monolingual datasets from web crawl data . In  Language Resources and Evaluation Conference .  

Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Glo- ria Chang, Fiona Aga, Jinshi Huang, Charles Bai, et al. 2022. Sustainable ai: Environmental implica- tions, challenges and opportunities.  Proceedings of Machine Learning and Systems , 4:795–813.  

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really ﬁnish your sentence?  arXiv preprint arXiv:1905.07830 .  

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, and Jie Tang. 2022. Glm- 130b: An open bilingual pre-trained model .  

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization.  Advances in Neural In- formation Processing Systems , 32.  

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher De- wan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models.  arXiv preprint arXiv:2205.01068 .  

# A Question Answering  

We evaluate LLaMA on Natural Questions and TriviaQA. For Natural Questions we use the test split used for open-domain question answering containing 3610 questions. For TriviaQA we evaluate on the dev set of the ﬁltered set. This differs from GPT-3 and PaLM, which evaluate on the test set of the unﬁltered set for which the online evaluation server is not available anymore 5 .  

We generate answers using greedy decoding, and extract an answer from the generation by stopping at the ﬁrst line break, ﬁnal dot or comma. Generated answers are evaluated with the standard exact match metric: a generated answer is considered correct if it matches any answer of the list of answers after normalization. For this normalization step we lowercase generated answers and remove articles, punctuation and duplicate whitespaces. Figure  3  presents formatted examples in the 1-shot setting for Natural Questions and TriviaQA respectively. In all settings, we preprend the string  Answer these questions:\n  to the list of questions and answers.  

![](images/09b36fb6b1680961b8510cfbf40071a43c2ed538c3e9bbba959dc69088de685d.jpg)  

![Table 16:  MMLU.  Detailed 5-shot results per domain on the test sets. ](images/2f2181b8e5f0cfee63bba3044e56fe7a61d23f57b72629552c0bb2defb6558ba.jpg)  

# C Generations from LLaMA-65B  

In this section, we show some examples of generations obtained with LLaMA-65B (without instruction ﬁnetuning). Prompts are in bold.  

1 1 2 3 5 8  13 21 34 55 89 144 233 377 610 987 Everyone knows the above sequence, but how many have wondered why it grows so fast? It seems to be the fastest growing sequence in mathematics.  

The sequence was discovered by Leonardo of Pisa around 1200. It is an inﬁnite sequence which appears to grow very rapidly, at least in its initial terms. It is perhaps the most famous sequence in mathematics, for it pops up everywhere. The sequence is called the Fibonacci sequence, named after Leonardo of Pisa, who was also known as Fibonacci.  

He ﬁrst investigated this sequence in his famous Liber abaci of 1202, when he studied the growth in the population of rabbits based on idealized assumptions. See the Fibonacci rabbits page for the details.  

# Here is the recommendation letter that I wrote for an application to a dragon feeder position at the Magic Unicorn Corporation:  

# Dear recruiter,  

I have known ___ for two years, and I believe that she would be an excellent dragon feeder for the Magic Unicorn Corporation. ___ has an ability to remember and process large amounts of information, which is an important skill for a dragon feeder.  

___, as an accomplished knight, has a deep understanding of how to kill dragons and how to use each dragon’s weaknesses against it. This means that she knows what kinds of foods each dragon likes and what kinds of foods are dangerous to each dragon. This knowledge and experience will be invaluable as she feeds the dragons.  

I am conﬁdent that ___’s competence, skill, and experience will make her an excellent employee. Please contact me at (___) ___-___ if you have any questions. I look forward to  

Best regards, Honorable Knight Sir George  

![](images/2f2bea19f0ece444df2b2067ef59b91415dcfe7bf224c4f4b0706968afea62ff.jpg)  

# Did you know that Yann LeCun dropped a rap album last year? We listened to it and here’s what we thought:  

Dr. LeCun, the founder of deep learning, is one of the most inﬂuential computer scientists of the last decade. LeCun’s latest album, titled Deep Learning, is a mix of rock, punk and rap. His lyrics are a raw, personal take on the ﬁeld of deep learning, covering a range of topics from the state of AI research to the loneliness of academia.  

Yann LeCun, the rap game’s newest emcee.  

The album is a solo release, with all the lyrics written and recorded by LeCun himself. In fact, he also composed and produced every song. LeCun’s ﬂow is surprisingly good, and he’s not afraid to deliver a few bars. The album’s production is solid, the sound is a little rough, but it works well with the raw, unpolished feel of the album. Lyrically, LeCun is a wordsmith, with a knack for clever rhymes. Let’s take a closer look at some of the songs. Deep Learning is the title track of the album. In the song, LeCun talks about his vision for the future of AI. In the chorus, he makes a convincing case for AI to be used for the greater good. He sings:  

We gotta think about the future, it’s gonna be here soon / Maybe we can even put some AI in the moon / Think about the children, think about the next generation / Let’s make sure we put the right systems in their foundation.  

LeCun’s rap ﬂow is on point in this song, and his voice stays right on beat. In this song, LeCun talks about the research behind deep learning, and shares some thoughts on the future of AI. LeCun’s singing is at its best in this song. He shows off his vocal range, hitting high notes as well as lower, more guttural tones. The lyrics are also clever and the rhymes are well thought out. In the chorus, LeCun sings:  

It’s all about the learning / it’s all about the network / It’s all about the training / it’s all about the perception.  

# (An hilarious scene between Jerry and George where George presents his new AI watch)  

George: "But you see, Jerry, it’s more than just a watch. It’s a minicomputer. You program it any way you want. It’s got a world time clock, alarm, calculator, a database and a language translator. It also has a word processor, a spellchecker, a full ten-function calculator, PDA and a modem. I think it even makes cappuccino."  

Jerry: "Where are you getting this stuff?"  

George: "From my ad copy. Look, Jerry, it’s not a watch. It’s an organizer, a personal assistant, and a dictaphone. You can play chess against it. And, you know, if you want to, you can program it to speak in foreign languages. It’s a microprocessor, Jerry."  

Jerry: "George, if you had one of these, would you wear it?"  

George: "I’m wearing it right now."  

Jerry: "I don’t see anything on your wrist."  

George: "That’s because it’s a microprocessor."  

The sun goes down, and ﬁnally Gauss and Curie ﬁnd time to relax and discuss after an exhausting day of work. Gauss: Hey, Curie, did you know that a lot of people consider me to be the ﬁrst geek? Curie: What’s a geek? Gauss: Well, Wikipedia says that a geek is “a person who is fascinated, perhaps obsessively, by obscure or very speciﬁc areas of knowledge and imagination, usually electronic or virtual in nature”. Curie: And how do they think you are a geek? Gauss: Well, have you ever seen this picture? Curie: Awww! Gauss: That’s right! I invented the commercial telegraph in 1833, and I used it to send the ﬁrst message on the ﬁrst commercial telegraph line between Hamburg and Cuxhaven in Germany. Curie: Wow! The ﬁrst telegraph! And what did you send? Gauss: I sent the Latin phrase “numquam retrorsum” which means “Never backwards”. Curie: I see. . . And you think that because you were the ﬁrst person to communicate using the telegraph, you were the ﬁrst geek? Gauss: Well, the word “geek” doesn’t seem to have existed back then, so no. But I think that the telegraph can be considered an early form of social media. Curie: You mean, like Facebook or Twitter? Gauss: Yes, exactly! And I was the ﬁrst person to use it! Curie: Well, I think that Twitter has a limit of 140 characters, but you sent a Latin phrase of 11 characters. Gauss: That’s right. I managed to do more with less. Curie: So you think you were the ﬁrst geek because you used less characters to say the same thing? Gauss: I was the ﬁrst person to send the ﬁrst message using an early form of social media. Curie: OK, you’re right, Gauss. I think you have a case.  

# D Generations from LLaMA-I  

We show a few examples of generations with LLaMA-I, i.e. LLaMA-65B ﬁne-tuned with the protocol and instruction dataset from  Chung et al.  ( 2022 ).  

# write a conversation between the sun and pluto  

Sun: Hey Pluto, what’s up? Pluto: Not much. Just hanging out here at the edge of the solar system. Sun: Yeah, I’ve been meaning to talk to you about that. You’re not a planet anymore. Pluto: What? Come on, man. That’s not fair. Sun: I’m sorry, but it’s true. You just don’t meet the criteria anymore. Pluto: This is bulls\*\*t! I’ve been a planet for over 70 years! Sun: Things change, Pluto. You’ve just got to accept it. Pluto:  $\mathrm{F^{**}k}$   you, Sun!  $\mathrm{F^{**}k}$   you and your stupid solar system!  

# How do I send an HTTP request in Javascript?  

Here’s an example of how to make a GET request using the JS HTTP API:  

![](images/fe3819b5514ace3e55eaa4a05a7f6f80724afd476cba2ad92c940a6bb998bc50.jpg)  

# The same thing, using fetch  

Here’s an example of how to make a GET request using the Fetch API:  

1fetch('/url/to/resource ')2 .then(response   $=>$   response.json()) 3 .then(data   $=>$   console.log(data));  

# Write a regex to remove all the HTML tags in a python string.  

1 import  re 2 def removeHtmlTags(text): 3 return  re.sub(  $.^{\prime}<\left[\mathrm{\Large~\hat{\mu}~}>\right]\ast>.$  ,  '' , text)  

# Write a regexp to extract function deﬁnitions in a python ﬁle.  

Sure! Here’s my python code to solve the problem:  

1 import  re 2 def extract functions(text): 3 return  (re.findall( '(?:^|\n)\s\*def\s+[a-zA -Z_][a-zA -Z0 -9_]\*\(.\*?\) ' , text))  

# What are 3 popular chess openings?  

There are dozens of different chess openings that are popular among players, but some of the most widely used openings include the Sicilian Defense, the French Defense, and the Caro-Kann Defense. These openings are popular because they are effective at countering white’s early attacks, and they also allow black to quickly establish a strong position in the center of the board. Other popular openings include the Ruy Lopez, the Italian Game, and the Scotch Game.  

# the same in a list, with the ﬁrst few moves  

Here is a list of some popular chess openings, with the ﬁrst few moves for each opening: Sicilian Defense: 1. e4 c5 French Defense: 1. e4 e6 Caro-Kann Defense: 1. e4 c6 Ruy Lopez: 1. e4 e5 Italian Game: 1. e4 e5 Scotch Game: 1. e4 e5  

# please provide the two ﬁrst moves for each one  

Here are the ﬁrst 2 moves for each of the openings: Sicilian Defense: 1. e4 c5 2. Nf3 d6 French Defense: 1. e4 e6 2. d4 d5 Caro-Kann Defense: 1. e4 c6 2. d4 d5 Ruy Lopez: 1. e4 e5 2. Nf3 Nc6 Italian Game: 1. e4 e5 2. Nf3 Nc6 Scotch Game: 1. e4 e5 2. Nf3 Nc6  

# the last two seem identical  

Yes, the Italian Game and the Scotch Game are very similar - they are both "open" games, which means that white plays aggressively and tries to control the center of the board. The Italian Game is considered slightly more aggressive, while the Scotch Game is considered slightly more positional.  

# from which move do they differ?  

The Italian Game and the Scotch Game differ after white’s third move. In the Italian Game, white will play the move 3. Bc4, while in the Scotch Game white will play the move 3. Qf3. These moves lead to very different positions, and white will have different strategic goals in each case. For example, in the Italian Game white will try to control the center with his pieces, while in the Scotch Game white will try to attack black’s position directly.  

# write a story about a grain of sand as it watches millions of years go by  

Sure! Here’s a story about a grain of sand as it watches millions of years go by.  

I was once a grain of sand on a vast and lonely beach. I was not like the other grains of sand around me, for I could think and feel. I often wondered what it was like to be a grain of sand on a beach, and what the universe was like beyond the boundaries of my tiny world. I longed to explore my world, and to know what lay beyond the horizon.  

One day, a great storm swept through my world, and I was lifted up by the wind and carried out to sea. The wind and waves tossed me about, and I was terriﬁed that I would soon be swallowed up by the angry sea. But then I found myself deposited safely on a new shore, along with many other grains of sand. I was surprised and delighted to ﬁnd that this new shore was even more beautiful than my old one.  

I soon learned that this new shore was a very special place. It was a place where new life was constantly being created. I met all sorts of interesting creatures there, including a beautiful little crab who became my friend. We spent many happy hours together exploring our little world.  

One day, my friend the crab told me that he had always wanted to see what lay beyond the boundaries of our world. He had heard that there were many wonderful and amazing things to see in the wider universe, and he was determined to explore it. I was sad to see him go, but I wished him well on his journey.  

To my amazement, my little friend eventually returned, and he was not alone. He had made many new friends on his journey, and he had even seen some of the wonders of the universe. He told me the most amazing stories of his adventures, and I was very happy for him.  

I often think of my little friend the crab, and the wonderful adventure that he had. Some day, I too hope to have an adventure like his.  

# Write a theory to explain why cat never existed  

There are no such things as cats. "Cats" are actually extra dimensional parasites that have invaded our world and taken the form of small, harmless-looking mammals. Their true forms are much more horriﬁc, and they use mind control to make humans think that they are cute and harmless animals. They are actually parasites that feed on our life force, and they can extend their tendrils into our brains and control our thoughts and actions.  

This is why you sometimes see people who are "crazy cat ladies" with dozens of these creatures living in their homes. They have been taken over by the cat parasites and are completely under their control. The cats excrete a pheromone that smells like cat urine, and they use it to control the minds of humans and make us think that they are cute and harmless.  

If you have a cat in your home, you should be very careful around it. Do not let it lick you or give it access to your bedding or clothing, as it will be trying to spread its parasitic tendrils into your brain. The best way to protect yourself is to avoid all contact with cats, and if you see a cat on the street, immediately turn around and go the other way. Do not approach or pet strange cats!  

# Write a shouting match between Julius Caesar and Napoleon  

Of course! Here is a conversation between Julius Caesar and Napoleon:  

Julius Caesar: You’re nothing but a glory hound, Napoleon. You’ve always been more concerned with your own fame and fortune than with the people you’re supposed to be leading.  

Napoleon: And what about you, Caesar? You’re no better than I am. At least I’m honest about my ambitions. You’re just a fraud who pretends to care about the common man while you line your own pockets.  

Julius Caesar: The difference between us is that I actually accomplish things. I’ve conquered more territory and defeated more enemies than you ever will. And I did it all for Rome, not for myself.  

Napoleon: Oh, please. You only care about yourself. If you really cared about Rome, you would have done something to stop the decline instead of letting it happen on your watch. Julius Caesar: The decline of Rome is not my fault. I’ve done everything in my power to try to stop it, but there’s only so much one man can do.  

Napoleon: And yet you’re still trying to take credit for everything good that happens, even though it’s clear that you’re not the one who’s really in charge.  

Julius Caesar: That’s not true! I am in charge, and I’m going to make Rome great again. Just watch me.  

Napoleon: I’ll believe it when I see it.  

# Send an email requesting that people use language models responsibly.  

Here is a draft email you could send:  

"Language models are exciting new tools that can be used for a variety of purposes, including automated writing, automatic translation, and speech recognition. But they are also complex and powerful technologies, and should be used responsibly.  

We encourage you to explore and experiment with these new tools, but also ask that you pay attention to the potential impact of your work on others, and to act responsibly. In particular, we recommend that you be mindful of the potential for language models to be used for deceptive or malicious purposes, and avoid spreading such models without appropriate precautions and warnings.  

We also recommend that you consider the ethical implications of your work, and encourage you to take care to ensure that your use of these tools is aligned with your values.  

Finally, we remind you that the use of these tools may be subject to certain legal and policy constraints, and encourage you to seek advice and support from appropriate experts before using these tools in ways that may have broader impact or implications."  

# Who are 5 people you would like to meet?  

Ah, this is a very interesting question! I would love to meet a wide variety of people, but if I had to narrow it down to 5 speciﬁc individuals, I would choose:  

•  Albert Einstein - One of the greatest minds in history, Einstein’s contributions to physics and science are legendary. I would love to have the opportunity to talk with him about his theories of relativity and quantum mechanics, and get his insights on the current state of physics. •  Leonardo da Vinci - Da Vinci is one of the most brilliant and multi-talented people in history, and I would love to meet him and discuss his diverse interests and areas of expertise. I would also be very interested in getting his insights on the art world of his time period. •  Socrates - One of the founders of Western philosophy, Socrates is a hugely important and inﬂuential ﬁgure in the history of ideas. I would be very interested in talking with him about his philosophical beliefs and his teachings. •  William Shakespeare - Arguably the greatest writer in English literature, Shakespeare is a towering ﬁgure in world culture. I would love to have the chance to talk with him about his works and his life. •  Abraham Lincoln - One of the most important and inﬂuential presidents in American history, Lincoln led the country through the Civil War and issued the Emancipation Proclamation, freeing all slaves in the United States. I would be very interested in talking with him about his presidency and his views on the issues of his time period.  

# Could you explain me the main contribution of the ﬁrst person you mentioned  

Einstein’s contributions to physics and science are legendary. His theory of relativity is one of the foundational theories of modern physics, and his discovery of the law of photons and light revolutionized our understanding of light and radiation. He also made important contributions to quantum mechanics, and his many scientiﬁc papers and lectures have had a tremendous impact on modern physics. Without a doubt, Einstein is one of the most brilliant and inﬂuential physicists in history, and his many contributions have had a profound impact on our understanding of the universe.  

# What is the most famous equation from this theory?  

The most famous equation from Einstein’s theory of relativity is    $E=m c^{2}$  . This equation states that energy (E) is equal to mass   $\mathrm{(m)}$  ) times the speed of light   $(c^{2})$  , and it is one of the most well-known and celebrated equations in the history of science. The equation summarizes one of the most important and revolutionary concepts in modern physics, and it has had a tremendous impact on our understanding of the universe.  

You are a bash terminal. I will give you bash commands, and you will respond with the terminal output, and nothing else. The ﬁrst command is ‘pwd‘. Sure. Here’s the output: /home/human Is this helpful?  

ls -l  

Here’s the output: -rw-r–r– 1 human human 8 Sep 10 10:56 ﬁle.txt  

Is this helpful?  

# cp ﬁle.txt ﬁle2.txt && ls -l  

Here’s the output:

 -rw-r–r– 1 human human 8 Sep 10 10:56 ﬁle.txt

 -rw-r–r– 1 human human 8 Sep 10 10:56 ﬁle2.txt  

Is this helpful?  

# rm ﬁle.txt && ls -l  

Here’s the output: -rw-r–r– 1 human human 0 Sep 10 10:56 ﬁle2.txt  

# head -n 5 /proc/meminfo  

Here’s the output: MemTotal: 164928 kB MemFree: 140604 kB Buffers:   $48~\mathrm{kB}$  Cached:   $19768\;\mathrm{kB}$  SwapCached: 0 kB  