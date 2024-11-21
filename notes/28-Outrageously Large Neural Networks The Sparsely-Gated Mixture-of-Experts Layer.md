# O UTRAGEOUSLY  L ARGE  N EURAL  N ETWORKS : T HE  S PARSELY -G ATED  M IXTURE - OF -E XPERTS  L AYER  

Noam Shazeer 1 , Azalia Mirhoseini ∗ † 1 , Krzysztof Maziarz ∗ 2 , Andy Davis 1 , Quoc  $\mathrm{Le^{1}}$  , Geoffrey Hinton 1   and Jeff Dean 1  

Google Brain, {noam,azalia,andydavis,qvl,geoffhinton,jeff}  $@$  google.com 2 Jagiellonian University, Cracow, krzysztof.maziarz  $@$  student.uj.edu.pl  

# A BSTRACT  

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increas- ing model capacity without a proportional increase in computation. In practice, however, there are signiﬁcant algorithmic and performance challenges. In this work, we address these challenges and ﬁnally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efﬁciency on modern GPU clusters. We in- troduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve signiﬁcantly better results than state-of-the-art at lower computational cost.  

# 1 I NTRODUCTION AND  R ELATED  W ORK  

# 1.1 C ONDITIONAL  C OMPUTATION  

Exploiting scale in both training data and model size has been central to the success of deep learn- ing. When datasets are sufﬁciently large, increasing the capacity (number of parameters) of neural networks can give much better prediction accuracy. This has been shown in domains such as text

 (Sutskever et al., 2014; Bahdanau et al., 2014; Jozefowicz et al., 2016; Wu et al., 2016), images

 (Krizhevsky et al., 2012; Le et al., 2012), and audio (Hinton et al., 2012; Amodei et al., 2015). For typical deep learning models, where the entire model is activated for every example, this leads to a roughly quadratic blow-up in training costs, as both the model size and the number of training examples increase. Unfortunately, the advances in computing power and distributed computation fall short of meeting such demand.  

Various forms of conditional computation have been proposed as a way to increase model capacity without a proportional increase in computational costs (Davis & Arel, 2013; Bengio et al., 2013; Eigen et al., 2013; Ludovic Denoyer, 2014; Cho & Bengio, 2014; Bengio et al., 2015; Almahairi et al., 2015). In these schemes, large parts of a network are active or inactive on a per-example basis. The gating decisions may be binary or sparse and continuous, stochastic or deterministic. Various forms of reinforcement learning and back-propagation are proposed for trarining the gating decisions.  

![](images/80e2c9ac64f57be0b4f47ddb31f570f1d1c0cc97be0e224dc3df0773b3363142.jpg)  
Figure 1: A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network.  

While these ideas are promising in theory, no work to date has yet demonstrated massive improve- ments in model capacity, training time, or model quality. We blame this on a combination of the following challenges:  

•  Modern computing devices, especially GPUs, are much faster at arithmetic than at branch- ing. Most of the works above recognize this and propose turning on/off large chunks of the network with each gating decision. •  Large batch sizes are critical for performance, as they amortize the costs of parameter trans- fers and updates. Conditional computation reduces the batch sizes for the conditionally active chunks of the network. •  Network bandwidth can be a bottleneck. A cluster of GPUs may have computational power thousands of times greater than the aggregate inter-device network bandwidth. To be com- putationally efﬁcient, the relative computational versus network demands of an algorithm must exceed this ratio. Embedding layers, which can be seen as a form of conditional com- putation, are handicapped by this very problem. Since the embeddings generally need to be sent across the network, the number of (example, parameter) interactions is limited by network bandwidth instead of computational capacity. •  Depending on the scheme, loss terms may be necessary to achieve the desired level of sparsity per-chunk and/or per example. Bengio et al. (2015) use three such terms. These issues can affect both model quality and load-balancing. •  Model capacity is most critical for very large data sets. The existing literature on condi- tional computation deals with relatively small image recognition data sets consisting of up to 600,000 images. It is hard to imagine that the labels of these images provide a sufﬁcient signal to adequately train a model with millions, let alone billions of parameters.  

In this work, we for the ﬁrst time address all of the above challenges and ﬁnally realize the promise of conditional computation. We obtain greater than   $1000\mathbf{x}$   improvements in model capacity with only minor losses in computational efﬁciency and signiﬁcantly advance the state-of-the-art results on public language modeling and translation data sets.  

# 1.2 O UR  A PPROACH : T HE  S PARSELY -G ATED  M IXTURE - OF -E XPERTS  L AYER  

Our approach to conditional computation is to introduce a new type of general purpose neural net- work component: a Sparsely-Gated Mixture-of-Experts Layer (MoE). The MoE consists of a num- ber of experts, each a simple feed-forward neural network, and a trainable gating network which selects a sparse combination of the experts to process each input (see Figure 1). All parts of the network are trained jointly by back-propagation.  

While the introduced technique is generic, in this paper we focus on language modeling and machine translation tasks, which are known to beneﬁt from very large models. In particular, we apply a MoE convolutionally between stacked LSTM layers (Hochreiter & Schmidhuber, 1997), as in Figure 1. The MoE is called once for each position in the text, selecting a potentially different combination of experts at each position. The different experts tend to become highly specialized based on syntax and semantics (see Appendix E Table 9). On both language modeling and machine translation benchmarks, we improve on best published results at a fraction of the computational cost.  

# 1.3 R ELATED WORK ON  M IXTURES OF  E XPERTS  

Since its introduction more than two decades ago (Jacobs et al., 1991; Jordan & Jacobs, 1994), the mixture-of-experts approach has been the subject of much research. Different types of expert architectures hae been proposed such as SVMs (Collobert et al., 2002), Gaussian Processes (Tresp, 2001; Theis & Bethge, 2015; Deisenroth & Ng, 2015), Dirichlet Processes (Shahbaba & Neal, 2009), and deep networks. Other work has focused on different expert conﬁgurations such as a hierarchical structure (Yao et al., 2009), inﬁnite numbers of experts (Rasmussen & Ghahramani, 2002), and adding experts sequentially (Aljundi et al., 2016). Garmash & Monz (2016) suggest an ensemble model in the format of mixture of experts for machine translation. The gating network is trained on a pre-trained ensemble NMT model.  

The works above concern top-level mixtures of experts. The mixture of experts is the whole model. Eigen et al. (2013) introduce the idea of using multiple MoEs with their own gating networks as parts of a deep model. It is intuitive that the latter approach is more powerful, since complex prob- lems may contain many sub-problems each requiring different experts. They also allude in their conclusion to the potential to introduce sparsity, turning MoEs into a vehicle for computational computation.  

Our work builds on this use of MoEs as a general purpose neural network component. While Eigen et al. (2013) uses two stacked MoEs allowing for two sets of gating decisions, our convolutional application of the MoE allows for different gating decisions at each position in the text. We also realize sparse gating and demonstrate its use as a practical way to massively increase model capacity.  

# 2 T HE  S TRUCTURE OF THE  M IXTURE - OF -E XPERTS LAYER  

The Mixture-of-E erts (MoE) layer consists f a set of    $n$   “expert networks"    $E_{1},\cdot\cdot\cdot\ ,E_{n}$  , and a “gating network"  G  whose output is a sparse  n -dimensional vector. Figure 1 shows an overview of the MoE module. The experts are themselves neural networks, each with their own parameters. Although in principle we only require that the experts accept the same sized inputs and produce the same-sized outputs, in our initial investigations in this paper, we restrict ourselves to the case where the models are feed-forward networks with identical architectures, but with separate parameters.  

Let us denote by    $G(x)$   and    $E_{i}(x)$   the output of the gating network and the output of the    $i$  -th expert network for a given input    $x$  . The output  $y$   of the MoE module can be written as follows:  

$$
y=\sum_{i=1}^{n}G(x)_{i}E_{i}(x)
$$  

We save computation based on the sparsity of the output of  $G(x)$  . Wherever  $G(x)_{i}=0$  , we need not compute    $E_{i}(x)$  . In our experiments, we have up to thousands of experts, but only need to evaluate a handful of them for every example. If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of “experts", each of which is itself a secondary mixture-of-experts with its own gating network. In the following we focus on ordinary MoEs. We provide more details on hierarchical MoEs in Appendix B.  

Our implementation is related to other models of conditional computation. A MoE whose experts are simple weight matrices is similar to the parameterized weight matrix proposed in (Cho & Bengio, 2014). A MoE whose experts have one hidden layer is similar to the block-wise dropout described in (Bengio et al., 2015), where the dropped-out layer is sandwiched between fully-activated layers.  

2.1 G ATING  N ETWORK  

Softmax Gating: A simple choice of non-sparse gating function (Jordan & Jacobs, 1994) is to multiply the input by a trainable weight matrix  $W_{g}$   and then apply the  Softmax  function.  

$$
G_{\sigma}(x)=S o f t m a x(x\cdot W_{g})
$$  

Noisy Top-K Gating: We add two components to the Softmax gating network: sparsity and noise. Before taking the softmax function, we add tunable Gaussian noise, then keep only the top  $\mathbf{k}$   values, setting the rest to    $-\infty$  (which causes the corresponding gate values to equal  0 ). The sparsity serves to save computation, as described above. While this form of sparsity creates some theoretically scary discontinuities in the output of gating function, we have not yet observed this to be a problem in practice. The noise term helps with load balancing, as will be discussed in Appendix A. The amount of noise per component is controlled by a second trainable weight matrix  $W_{n o i s e}$  .  

$$
G(x)=S o f t m a x(K e e p T o p K(H(x),k))
$$  

$$
H(x)_{i}=(x\cdot W_{g})_{i}+S t a n d a r d N o r m a l()\cdot S o f t p l u s((x\cdot W_{n o i s e})_{i})
$$  

$$
K e e p T o p K(v,k)_{i}={\left\{\begin{array}{l l}{v_{i}}&{{\mathrm{if~}}v_{i}{\mathrm{~is~in~the~top~}}k{\mathrm{~elements~of~}}v.}\\ {-\infty}&{{\mathrm{otherwise}}.}\end{array}\right.}
$$  

Training the Gating Network We train the gating network by simple back-propagation, along with the rest of the model. If we choose    $k>1$  , the gate values for the top  $\mathbf{k}$   experts have nonzero derivatives with respect to the weights of the gating network. This type of occasionally-sensitive behavior is described in (Bengio et al., 2013) with respect to noisy rectiﬁers. Gradients also back- propagate through the gating network to its inputs. Our method differs here from (Bengio et al., 2015) who use boolean gates and a REINFORCE-style approach to train the gating network.  

3 A DDRESSING  P ERFORMANCE  C HALLENGES 3.1 T HE  S HRINKING  B ATCH  P ROBLEM  

On modern CPUs and GPUs, large batch sizes are necessary for computational efﬁciency, so as to amortize the overhead of parameter loads and updates. If the gating network chooses  $k$   out of  $n$   experts for each example, then for a batch of    $b$   examples, each expert receives a much smaller batch of approximately very inefﬁcient as the number of experts increases. The solution to this shrinking batch problem is    $\begin{array}{r}{\frac{\d^{k}b}{n}\,\ll\,b\qquad}\end{array}$     examples. This causes a naive MoE implementation to become to make the original batch size as large as possible. However, batch size tends to be limited by the memory necessary to store activations between the forwards and backwards passes. We propose the following techniques for increasing the batch size:  

Mixing Data Parallelism and Model Parallelism: In a conventional distributed training setting, multiple copies of the model on different devices asynchronously process distinct batches of data, and parameters are synchronized through a set of parameter servers. In our technique, these different batches run synchronously so that they can be combined for the MoE layer. We distribute the standard layers of the model and the gating network according to conventional data-parallel schemes, but keep only one shared copy of each expert. Each expert in the MoE layer receives a combined batch consisting of the relevant examples from all of the data-parallel input batches. The same set of devices function as data-parallel replicas (for the standard layers and the gating networks) and as model-parallel shards (each hosting a subset of the experts). If the model is distributed over  $d$  devices, and each device processes a batch of size    $b$  , each expert receives a batch of approximately  $\frac{k b d}{n}$    examples. Thus, we achieve a factor of    $d$   improvement in expert batch size.  

In the case of a hierarchical MoE (Section B), the primary gating network employs data parallelism, and the secondary MoEs employ model parallelism. Each secondary MoE resides on one device.  

This technique allows us to increase the number of experts (and hence the number of parameters) by proportionally increasing the number of devices in the training cluster. The total batch size increases, keeping the batch size per expert constant. The memory and bandwidth requirements per device also remain constant, as do the step times, as does the amount of time necessary to process a number of training examples equal to the number of parameters in the model. It is our goal to train a trillion- parameter model on a trillion-word corpus. We have not scaled our systems this far as of the writing of this paper, but it should be possible by adding more hardware.  

Taking Advantage of Convolutional it y: In our language models, we apply the same MoE to each time step of the previous layer. If we wait for the previous layer to ﬁnish, we can apply the MoE to all the time steps together as one big batch. Doing so increases the size of the input batch to the MoE layer by a factor of the number of unrolled time steps.  

Increasing Batch Size for a Recurrent MoE: We suspect that even more powerful models may involve applying a MoE recurrently. For example, the weight matrices of a LSTM or other RNN could be replaced by a MoE. Sadly, such models break the convolutional trick from the last para- graph, since the input to the MoE at one timestep depends on the output of the MoE at the previous timestep. Gruslys et al. (2016) describe a technique for drastically reducing the number of stored activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow for a large increase in batch size.  

# 3.2 N ETWORK  B ANDWIDTH  

Another major performance concern in distributed computing is network bandwidth. Since the ex- perts are stationary (see above) and the number of gating parameters is small, most of the communi- cation involves sending the inputs and outputs of the experts across the network. To maintain com- putational efﬁciency, the ratio of an expert’s computation to the size of its input and output must ex- ceed the ratio of computational to network capacity of the computing device. For GPUs, this may be thousands to one. In our experiments, we use experts with one hidden layer containing thousands of RELU-activated units. Since the weight matrices in the expert have sizes  input _ size × hidden _ size and  hidden _ size  ×  output _ size , the ratio of computation to input and output is equal to the size of the hidden layer. Conveniently, we can increase computational efﬁciency simply by using a larger hidden layer, or more hidden layers.  

# 4 B ALANCING  E XPERT  U TILIZATION  

We have observed that the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network. Eigen et al. (2013) describe the same phenomenon, and use a hard constraint at the beginning of training to avoid this local minimum. Bengio et al. (2015) include a soft constraint on the batch-wise average of each gate.  

We take a soft constraint approach. We deﬁne the importance of an expert relative to a batch of training examples to be the batchwise sum of the gate values for that expert. We deﬁne an additional loss    $L_{i m p o r t a n c e}$  , which is added to the overall loss function for the model. This loss is equal to the square of the coefﬁcient of variation of the set of importance values, multiplied by a hand-tuned scaling factor  $w_{i m p o r t a n c e}$  . This additional loss encourages all experts to have equal importance.  

$$
I m p o r t a n c e(X)=\sum_{x\in X}G(x)
$$  

$$
L_{i m p o r t a n c e}(X)=w_{i m p o r t a n c e}\cdot C V(I m p o r t a n c e(X))^{2}
$$  

While this loss function can ensure equal importance, experts may still receive very different num- bers of examples. For example, one expert may receive a few examples with large weights, and another may receive many examples with small weights. This can cause memory and performance problems on distributed hardware. To solve this problem, we introduce a second loss function,  $L_{l o a d}$   , which ensures balanced loads. Appendix A contains the deﬁnition of this function, along with experimental results.  

# 5 E XPERIMENTS  

# 5.1 1 B ILLION  W ORD  L ANGUAGE  M ODELING  B ENCHMARK  

Dataset: This dataset, introduced by (Chelba et al., 2013) consists of shufﬂed unique sentences from news articles, totaling approximately 829 million words, with a vocabulary of 793,471 words.  

Previous State-of-the-Art: The best previously published results (Jozefowicz et al., 2016) use models consisting of one or more stacked Long Short-Term Memory (LSTM) layers (Hochreiter & Schmidhuber, 1997; Gers et al., 2000). The number of parameters in the LSTM layers of these models vary from 2 million to 151 million. Quality increases greatly with parameter count, as do computational costs. Results for these models form the top line of Figure 2-right.  

MoE Models: Our models consist of two stacked LSTM layers with a MoE layer between them (see Figure 1). We vary the sizes of the layers and the number of experts. For full details on model architecture, training regimen, additional baselines and results, see Appendix C.  

Low Computation, Varied Capacity: To investigate the effects of adding capacity, we trained a series of MoE models all with roughly equal computational costs: about 8 million multiply-and- adds per training example per timestep in the forwards pass, excluding the softmax layer. We call this metric (ops/timestep). We trained models with ﬂat MoEs containing 4, 32, and 256 experts, and models with hierarchical MoEs containing 256, 1024, and 4096 experts. Each expert had about 1 million parameters. For all the MoE layers, 4 experts were active per input.  

The results of these models are shown in Figure 2-left. The model with 4 always-active experts performed (unsurprisingly) similarly to the computationally-matched baseline models, while the largest of the models (4096 experts) achieved an impressive  $24\%$   lower perplexity on the test set.  

![](images/159870a1e81330fb3812196c21f405c58b81920972d93f3c234d193ff2fa80fb.jpg)  
Figure 2: Model comparison on 1-Billion-Word Language-Modeling Benchmark. On the left, we plot test perplexity as a function of model capacity for models with similar computational budgets of approximately 8-million-ops-per-timestep. On the right, we plot test perplexity as a function of computational budget. The top line represents the LSTM models from (Jozefowicz et al., 2016). The bottom line represents 4-billion parameter MoE models with different computational budgets.  

Varied Computation, High Capacity: In addition to the largest model from the previous section, we trained two more MoE models with similarly high capacity (4 billion parameters), but higher computation budgets. These models had larger LSTMs, and fewer but larger and experts. Details  

![Table 1: Summary of high-capacity MoE-augmented models with varying computational budgets, vs. best previously published results (Jozefowicz et al., 2016). Details in Appendix C. ](images/f2726e0cdd099d946ad3057e25772f6d0273836187a89995755e1d1ed05a3c30.jpg)  

can be found in Appendix C.2. Results of these three models form the bottom line of Figure 2-right. Table 1 compares the results of these models to the best previously-published result on this dataset . Even the fastest of these models beats the best published result (when controlling for the number of training epochs), despite requiring only  $6\%$   of the computation.  

Computational Efﬁciency: We trained our models using TensorFlow (Abadi et al., 2016) on clus- ters containing 16-32 Tesla K40 GPUs. For each of our models, we determine computational efﬁ- ciency in TFLOPS/GPU by dividing the number of ﬂoating point operations required to process one training batch by the observed step time and the number of GPUs in the cluster. The operation counts used here are higher than the ones we report in our ops/timestep numbers in that we include the backwards pass, we include the importance-sampling-based training of the softmax layer, and we count a multiply-and-add as two separate operations. For all of our MoE models, the ﬂoating point operations involved in the experts represent between  $37\%$   and   $46\%$   of the total.  

For our baseline models wtih no MoE, observed computational efﬁciency ranged from 1.07-1.29 TFLOPS/GPU. For our low-computation MoE models, computation efﬁciency ranged from 0.74- 0.90 TFLOPS/GPU, except for the 4-expert model which did not make full use of the available parallelism. Our highest-computation MoE model was more efﬁcient at 1.56 TFLOPS/GPU, likely due to the larger matrices. These numbers represent a signiﬁcant fraction of the theoretical maximum of 4.29 TFLOPS/GPU claimed by NVIDIA. Detailed results are in Appendix C, Table 7.  

# 5.2 100 B ILLION  W ORD  G OOGLE  N EWS  C ORPUS  

![](images/1697fc330ca631a3aa2b293b57bf80a0260648ef6228a738c4d1f3aa5ef71bcd.jpg)  
Figure 3: Language modeling on a 100 billion word corpus. Models have similar computational budgets (8 million ops/timestep).  

On the 1-billion-word corpus, adding additional capacity seems to produce diminishing returns as the number of parameters in the MoE layer exceeds 1 billion, as can be seen in Figure 2-left. We hypothesized that for a larger training set, even higher capacities would produce signiﬁcant quality improvements.  

We constructed a similar training set consisting of shufﬂed unique sentences from Google’s internal news corpus, totalling roughly 100 billion words. Similarly to the previous section, we tested a series of models with similar computational costs of about 8 million ops/timestep. In addition to a baseline LSTM model, we trained models augmented with MoE layers containing 32, 256, 1024, 4096, 16384, 65536, and 131072 experts. This corresponds to up to 137 billion parameters in the MoE layer. Details on architecture, training, and results are given in Appendix D.  

Results: Figure 3 shows test perplexity as a function of capacity after training on 10 billion words (top line) and 100 billion words (bottom line). When training over the full 100 billion words, test perplexity improves signiﬁcantly up to 65536 experts (68 billion parameters), dropping  $39\%$   lower than the computationally matched baseline, but degrades at 131072 experts, possibly a result of too much sparsity. The widening gap between the two lines demonstrates (unsurprisingly) that increased model capacity helps more on larger training sets.  

Even at 65536 experts   $(99.994\%$   layer sparsity), computational efﬁciency for the model stays at a respectable 0.72 TFLOPS/GPU.  

5.3MACHINE TRANSLATION (SINGLE LANGUAGE PAIR)  

Model Architecture: Our model was a modiﬁed version of the GNMT model described in (Wu et al., 2016). To reduce computation, we decreased the number of LSTM layers in the encoder and decoder from 9 and 8 to 3 and 2 respectively. We inserted MoE layers in both the encoder (between layers 2 and 3) and the decoder (between layers 1 and 2). Each MoE layer contained up to 2048 experts each with about two million parameters, adding a total of about 8 billion parameters to the models. Further details on model architecture, testing procedure and results can be found in Appendix E.  

Datasets: We benchmarked our method on the WMT’14 En  ${\rightarrow}\mathrm{F_{l}}$  r and   $\mathrm{En}{\rightarrow}\mathrm{De}$   corpora, whose training sets have 36M sentence pairs and 5M sentence pairs, respectively. The experimental proto- cols were also similar to those in (Wu et al., 2016): newstest2014 was used as the test set to compare against previous work (Luong et al., 2015a; Zhou et al., 2016; Wu et al., 2016), while the combina- tion of newstest2012 and newstest2013 was used as the development set. We also tested the same model on a Google’s Production English to French data.  

![Table 2: Results on WMT’  $14\,\mathrm{En}\rightarrow$  Fr newstest2014 (bold values represent best results). ](images/5046e4f91a9f212e7766f1dae8631465785d5e2da2cb18ef3d4582f7594c468b.jpg)  

![Table 3: Results on WMT’14 En  $\rightarrow$  De newstest2014 (bold values represent best results). ](images/79eccc3209d1284881a296b8156b42d3dbae05d98d2d73d0f4b0558b26552e43.jpg)  

![Table 4: Results on the Google Production  $\mathrm{En}{\rightarrow}$  Fr dataset (bold values represent best results). ](images/e3e1cf093bd96cdce65562a2239fab12c1bcfca8504bd356e6c98aed079a10cf.jpg)  

Results: Tables 2, 3, and 4 show the results of our largest models, compared with published Our approach achieved BLEU scores of 40.56 and 26.03 on the WMT’  ${14\ \mathrm{En}{\rightarrow}\mathrm{Fr}}$   and  $\mathrm{En}{\rightarrow}\mathrm{De}$  → De benchmarks. As our models did not use RL reﬁnement, these results constitute signiﬁcant gains of 1.34 and 1.12 BLEU score on top of the strong baselines in (Wu et al., 2016). The perplexity scores are also better.   On the Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time.  

# 5.4 M ULTILINGUAL  M ACHINE  T RANSLATION  

Dataset: (Johnson et al., 2016) train a single GNMT (Wu et al., 2016) model on a very large com- bined dataset of twelve language pairs. Results are somewhat worse than those for 12 separately trained single-pair GNMT models. This is not surprising, given that the twelve models have 12 times the capacity and twelve times the aggregate training of the one model. We repeat this ex- periment with a single MoE-augmented model. See Appendix E for details on model architecture. We train our model on the same dataset as (Johnson et al., 2016) and process the same number of training examples (about 3 billion sentence pairs). Our training time was shorter due to the lower computational budget of our model.  

Results: Results for the single-pair GNMT models, the multilingual GNMT model and the mul- tilingual MoE model are given in Table 5. The MoE model achieves   $19\%$   lower perplexity on the dev set than the multilingual GNMT model. On BLEU score, the MoE model signiﬁcantly beats the multilingual GNMT model on 11 of the 12 language pairs (by as much as 5.84 points), and even beats the monolingual GNMT models on 8 of 12 language pairs. The poor performance on English  $\rightarrow$  Korean seems to be a result of severe overtraining, as for the rarer language pairs a small number of real examples were highly oversampled in the training corpus.  

![Table 5: Multilingual Machine Translation (bold values represent best results). ](images/ca1a82c4cc226bddae417f5392e98ad4b0bde03e916e93e3d95a32a4a9c628c8.jpg)  

# 6 C ONCLUSION  

This work is the ﬁrst to demonstrate major wins from conditional computation in deep networks. We carefully identiﬁed the design considerations and challenges of conditional computing and ad- dressed them with a combination of algorithmic and engineering solutions. While we focused on text, conditional computation may help in other domains as well, provided sufﬁciently large train- ing sets. We look forward to seeing many novel implementations and applications of conditional computation in the years to come.  

A CKNOWLEDGMENTS  

We would like to thank all of the members of the Google Brain and Google Translate teams who helped us with this project, in particular Zhifeng Chen, Yonghui Wu, and Melvin Johnson. Thanks also to our anonymous ICLR reviewers for the helpful suggestions on making this paper better.  

# R EFERENCES  

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Gre- gory S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian J. Good- fellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Józefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Gor- don Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul A. Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda B. Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. Tensorﬂow: Large-scale machine learning on heterogeneous distributed systems. CoRR , abs/1603.04467, 2016. URL  http://arxiv.org/abs/1603.04467 .  

Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts.  CoRR , abs/1611.06194, 2016. URL  http://arxiv.org/abs/1611. 06194 .  

A. Almahairi, N. Ballas, T. Cooijmans, Y. Zheng, H. Larochelle, and A. Courville. Dynamic Capac- ity Networks.  ArXiv e-prints , November 2015.  

Dario Amodei, Rishita Anubhai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Jing- dong Chen, Mike Chrzanowski, Adam Coates, Greg Diamos, Erich Elsen, Jesse Engel, Linxi Fan, Christopher Fougner, Tony Han, Awni Y. Hannun, Billy Jun, Patrick LeGresley, Libby Lin, Sharan Narang, Andrew Y. Ng, Sherjil Ozair, Ryan Prenger, Jonathan Raiman, Sanjeev Satheesh, David Seetapun, Shubho Sengupta, Yi Wang, Zhiqian Wang, Chong Wang, Bo Xiao, Dani Yo- gatama, Jun Zhan, and Zhenyao Zhu. Deep speech 2: End-to-end speech recognition in english and mandarin.  arXiv preprint arXiv:1512.02595 , 2015.  

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate.  arXiv preprint arXiv:1409.0473 , 2014.  

Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models.  arXiv preprint arXiv:1511.06297 , 2015.  

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation.  arXiv preprint arXiv:1308.3432 , 2013.  

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005 , 2013.  

K. Cho and Y. Bengio. Exponentially Increasing the Capacity-to-Computation Ratio for Conditional Computation in Deep Learning.  ArXiv e-prints , June 2014.  

Ronan Collobert, Samy Bengio, and Yoshua Bengio. A parallel mixture of SVMs for very large scale problems.  Neural Computing , 2002.  

Andrew Davis and Itamar Arel. Low-rank approximations for conditional feedforward computation in deep neural networks.  arXiv preprint arXiv:1312.4461 , 2013.  

Marc Peter Deisenroth and Jun Wei Ng. Distributed Gaussian processes. In  ICML , 2015.  

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization, 2010.  

Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heaﬁeld. Edinburgh’s phrase-based machine translation systems for wmt-14. In  Proceedings of the Ninth Workshop on Statistical Machine Translation , 2014.  

David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. Learning factored representations in a deep mixture of experts.  arXiv preprint arXiv:1312.4314 , 2013.  

Felix A. Gers, Jürgen A. Schmidhuber, and Fred A. Cummins. Learning to forget: Continual pre- diction with lstm.  Neural Computation , 2000. Audrunas Gruslys, Rémi Munos, Ivo Danihelka, Marc Lanctot, and Alex Graves. Memory-efﬁcient backpropagation through time.  CoRR , abs/1606.03401, 2016. URL  http://arxiv.org/ abs/1606.03401 . Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog- nition.  IEEE Conference on Computer Vision and Pattern Recognition , 2015. Geoffrey Hinton, Li Deng, Dong Yu, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups.  IEEE Signal Processing Magazine , 2012. Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory.  Neural Computation , 1997. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift.  arXiv preprint arXiv:1502.03167 , 2015. Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts.  Neural Computing , 1991. Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda B. Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s multilingual neural machine translation system: Enabling zero-shot translation. CoRR , abs/1611.04558, 2016. URL  http://arxiv.org/abs/1611.04558 . Michael I. Jordan and Robert A. Jacobs. Hierarchical mixtures of experts and the EM algorithm. Neural Computing , 1994. Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling.  arXiv preprint arXiv:1602.02410 , 2016. Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In  ICLR , 2015. Reinhard Kneser and Hermann. Ney. Improved backingoff for m-gram language modeling., 1995. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classiﬁcation with deep convo- lutional neural networks. In  NIPS , 2012. Quoc V. Le, Marc’Aurelio Ranzato, Rajat Monga, Matthieu Devin, Kai Chen, Greg S. Corrado, Jeffrey Dean, and Andrew Y. Ng. Building high-level features using large scale unsupervised learning. In  ICML , 2012. Patrick Gallinari Ludovic Denoyer. Deep sequential neural network. arXiv preprint arXiv:1410.0510 , 2014. Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention- based neural machine translation.  EMNLP , 2015a. Minh-Thang Luong, Ilya Sutskever, Quoc V. Le, Oriol Vinyals, and Wojciech Zaremba. Addressing the rare word problem in neural machine translation.  ACL , 2015b. Carl Edward Rasmussen and Zoubin Ghahramani. Inﬁnite mixtures of Gaussian process experts. NIPS , 2002. Hasim Sak, Andrew W Senior, and Françoise Beaufays. Long short-term memory recurrent neural network architectures for large scale acoustic modeling. In  INTERSPEECH , pp. 338–342, 2014. Mike Schuster and Kaisuke Nakajima. Japanese and Korean voice search.  ICASSP , 2012.  

2009.  

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In  NIPS , 2014.  

Lucas Theis and Matthias Bethge. Generative image modeling using spatial LSTMs. In  NIPS , 2015.  

Volker Tresp. Mixtures of Gaussian Processes. In  NIPS , 2001.  

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin John- son, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s neural machine translation system: Bridging the gap between human and machine translation.  arXiv preprint arXiv:1609.08144 , 2016.  

Bangpeng Yao, Dirk Walther, Diane Beck, and Li Fei-fei. Hierarchical mixture of classiﬁcation experts uncovers interactions between brain regions. In  NIPS . 2009.  

Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329 , 2014.  

Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation.  arXiv preprint arXiv:1606.04199 , 2016.  

# A PPENDICES  

# A L OAD -B ALANCING  L OSS  

As discussed in section 4, for load-balancing purposes, we want to deﬁne an additional loss function to encourage experts to receive roughly equal numbers of training examples. Unfortunately, the number of examples received by an expert is a discrete quantity, so it can not be used in back- propagation. Instead, we deﬁne a smooth estimator    $L o a d(X)$   of the number of examples assigned to each expert for a batch    $X$   of inputs. The smoothness allows us to back-propagate gradients through the estimator. This is the purpose of the noise term in the gating function. We deﬁne    $P(x,i)$   as the probability that    $G(x)_{i}$   is nonzero, given a new random choice of noise on element    $i$  , but keeping the already-sampled choices of noise on the other elements. To compute  $P(x,i)$  , we note that the  $G(x)_{i}$   is nonzero if and only if    $H(x)_{i}$   is greater than the    $k^{t h}$  -greatest element of    $H(x)$   excluding itself. The probability works out to be:  

$$
\begin{array}{r}{P(x,i)=P r\Big((x\cdot W_{g})_{i}+S t a n d a r d N o r m a l()\cdot S o f t p l u s((x\cdot W_{n o i s e})_{i})}\\ {>k t h\_e x c l u d i n g(H(x),k,i)\Big)}\end{array}
$$  

Where  kth _ excluding  $(v,k,i)$   means the kth highest component of  $v$  , excluding component  $i$  . Sim- plifying, we get:  

$$
P(x,i)=\Phi\Big(\frac{(x\cdot W_{g})_{i}-k t h\_e x c l u d i n g(H(x),k,i)}{S o f t p l u s((x\cdot W_{n o i s e})_{i})}\Big)
$$  

Where    $\Phi$   is the CDF of the standard normal distribution.  

$$
L o a d(\boldsymbol{X})_{i}=\sum_{\boldsymbol{x}\in\boldsymbol{X}}P(\boldsymbol{x},i)
$$  

We can now deﬁne the load loss to be the square of the coefﬁcient of variation of the load vector, multiplied by a hand-tuned scaling factor  $w_{l o a d}$  .  

$$
L_{l o a d}(X)=w_{l o a d}\cdot C V(L o a d(X))^{2}
$$  

Initial Load Imbalance: To avoid out-of-memory errors, we need to initialize the network in a state of approximately equal expert load (since the soft constraints need some time to work). To accomplish this, we initialize the matrices    $W_{g}$   and    $W_{n o i s e}$   to all zeros, which yields no signal and some noise.  

Experiments: We trained a set of models with identical architecture (the MoE-256 model de- scribed in Appendix C), using different values of    $w_{i m p o r t a n c e}$   and  $w_{l o a d}$  . We trained each model for 10 epochs, then measured perplexity on the test set. We also measured the coefﬁcients of variation in  Importance  and  Load , as well as ratio of the load on the most overloaded expert to the average load. This last value is signiﬁcant for load balancing purposes on distributed hardware. All of these metrics were averaged over several training batches.  

![Table 6: Experiments with different combinations of losses. ](images/198eb1d86904d64b1b7383d9f966c8ddaf80c6e2e956a990c63aadc429745aad.jpg)  

Results: Results are reported in Table 6. All the combinations containing at least one the two losses led to very similar model quality, where having no loss was much worse. Models with higher values of  $w_{l o a d}$   had lower loads on the most overloaded expert.  

# B H IERACHICAL  M IXTURE OF  E XPERTS  

If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted com- bination of “experts", each of which is itself a secondary mixture-of-experts with its own gating network.   If the hierarchical MoE consists of  $a$   groups of    $b$   experts each, we denote the primary gat- ing network by  $G_{p r i m a r y}$  , the secondary gating networks by    $(G_{1},G_{2}..G_{a})$  , and the expert networks by    $\left(E_{0,0},E_{0,1}..E_{a,b}\right)$  . The output of the MoE is given by:  

$$
y_{H}=\sum_{i=1}^{a}\sum_{j=1}^{b}G_{p r i m a r y}(x)_{i}\cdot G_{i}(x)_{j}\cdot E_{i,j}(x)
$$  

Our metrics of expert utilization change to the following:  

$$
I m p o r t a n c e_{H}(X)_{i,j}=\sum_{x\in X}G_{p r i m a r y}(x)_{i}\cdot G_{i}(x)_{j}
$$  

$$
L o a d_{H}(X)_{i,j}=\frac{L o a d_{p r i m a r y}(X)_{i}\cdot L o a d_{i}(X^{(i)})_{j}}{|X^{(i)}|}
$$  

$L o a d_{p r i m a r y}$   and    $L o a d_{i}$   deonte the  Load  functions for the primary gating network and    $i^{t h}$    sec- ondary gating network respectively.    $X^{(i)}$    denotes the subset of    $X$   for which    $G_{p r i m a r y}(x)_{i}>0$  .  

It would seem simpler to let  $L o a d_{H}(X)_{i,j}=L o a d_{i}(X_{i})_{j}$   , but this would not have a gradient with respect to the primary gating network, so we use the formulation above.  

C 1 B ILLION  W ORD  L ANGUAGE  M ODELING  B ENCHMARK  - E XPERIMENTAL  D ETAILS  

C.18-MILLION-OPERATIONS-PER-TIMESTEP MODELS  

Model Architecture: Our model consists of ﬁve layers: a word embedding layer, a recurrent Long Short-Term Memory (LSTM) layer (Hochreiter   $\&$   Schmidhuber, 1997; Gers et al., 2000), a MoE layer, a second LSTM layer, and a softmax layer. The dimensionality of the embedding layer, the number of units in each LSTM layer, and the input and output dimensionality of the MoE layer are all equal to 512. For every layer other than the softmax, we apply drouput (Zaremba et al., 2014) to the layer output, dropping each activation with probability  DropProb , otherwise dividing by    $(1-D r o p P r o b)$  . After dropout, the output of the previous layer is added to the layer output. This residual connection encourages gradient ﬂow (He et al., 2015).  

MoE Layer Architecture: Each expert in the MoE layer is a feed forward network with one ReLU-activated hidden layer of size 1024 and an output layer of size 512. Thus, each expert contains  $[512*1024]+[1024*512]=1M$   parameters. The output of the MoE layer is passed through a sigmoid function before dropout. We varied the number of experts between models, using ordinary MoE layers with 4, 32 and 256 experts and hierarchical MoE layers with 256, 1024 and 4096 experts. We call the resulting models MoE-4, MoE-32, MoE-256, MoE-256-h, MoE-1024-h and MoE-4096- h. For the hierarchical MoE layers, the ﬁrst level branching factor was 16, corresponding to the number of GPUs in our cluster. We use Noisy-Top-K Gating (see Section 2.1) with  $k=4$   for the ordinary MoE layers and    $k=2$   at each level of the hierarchical MoE layers. Thus, each example is processed by exactly 4 experts for a total of 4M ops/timestep. The two LSTM layers contribute 2M ops/timestep each for the desired total of 8M.  

Computationally-Matched Baselines: The MoE-4 model does not employ sparsity, since all 4 experts are always used. In addition, we trained four more computationally-matched baseline models with no sparsity:  

•  MoE-1-Wide: The MoE layer consists of a single "expert" containing one ReLU-activated hidden layer of size 4096. •  MoE-1-Deep: The MoE layer consists of a single "expert" containing four ReLU-activated hidden layers, each with size  1024 . •  4xLSTM-512: We replace the MoE layer with two additional 512-unit LSTM layers. •  LSTM-2048-512: The model contains one 2048-unit LSTM layer (and no MoE). The out- put of the LSTM is projected down to 512 dimensions (Sak et al., 2014). The next timestep of the LSTM receives the projected output. This is identical to one of the models published in (Jozefowicz et al., 2016). We re-ran it to account for differences in training regimen, and obtained results very similar to the published ones.  

Training: The models were trained on a cluster of 16 K40 GPUs using the synchronous method described in Section 3. Each batch consisted of a set of sentences totaling roughly 300,000 words. In the interest of time, we limited training to 10 epochs, (27,000 steps). Training took 12-16 hours for all models, except for MoE-4, which took 18 hours (since all the expert computation was performed on only 4 of 16 GPUs). We used the Adam optimizer (Kingma & Ba, 2015). The base learning rate was increased linearly for the ﬁrst 1000 training steps, and decreased after that so as to be proportional to the inverse square root of the step number. The Softmax output layer was trained efﬁciently using importance sampling similarly to the models in (Jozefowicz et al., 2016). For each model, we performed a hyper-parmeter search to ﬁnd the best dropout probability, in increments of 0.1.  

To ensure balanced expert utilization we set    $w_{i m p o r t a n c e}\,=\,0.1$   and    $w_{l o a d}\,=\,0.1$  , as described in Section 4 and Appendix A.  

Results: We evaluate our model using perplexity on the holdout dataset, used by (Chelba et al., 2013; Jozefowicz et al., 2016). We follow the standard procedure and sum over all the words in- cluding the end of sentence symbol. Results are reported in Table 7. For each model, we report the test perplexity, the computational budget, the parameter counts, the value of  DropProb , and the computational efﬁciency.  

![Table 7: Model comparison on 1 Billion Word Language Modeling Benchmark. Models marked with \* are from (Jozefowicz et al., 2016). ](images/05f9fe146e3ee60951163c723a8cc777fb148c21380cbb874e62575d8dd3e2d6.jpg)  

We ran two additional models (MoE-34M and MoE-143M) to investigate the effects of adding more computation in the presence of a large MoE layer. These models have computation budgets of 34M and 143M ops/timestep. Similar to the models above, these models use a MoE layer between two LSTM layers. The dimensionality of the embedding layer, and the input and output dimensionality of the MoE layer are set to 1024 instead of 512. For MoE-34M, the LSTM layers have 1024 units. For MoE-143M, the LSTM layers have 4096 units and an output projection of size 1024 (Sak et al., 2014). MoE-34M uses a hierarchical MoE layer with 1024 experts, each with a hidden layer of size

 2048. MoE-143M uses a hierarchical MoE layer with 256 experts, each with a hidden layer of size

 8192. Both models have 4B parameters in the MoE layers. We searched for the best  DropProb  for each model, and trained each model for 10 epochs.  

The two models achieved test perplexity of  31 . 3  and  28 . 0  respectively, showing that even in the presence of a large MoE, more computation is still useful. Results are reported at the bottom of Table 7. The larger of the two models has a similar computational budget to the best published model from the literature, and training times are similar. Comparing after 10 epochs, our model has a lower test perplexity by  $18\%$  .  

# D 100 B ILLION  W ORD  G OOGLE  N EWS  C ORPUS  - E XPERIMENTAL  D ETAILS  

Model Architecture: The models are similar in structure to the 8-million-operations-per-timestep models described in the previous section. We vary the number of experts between models, using an ordinary MoE layer with 32 experts and hierarchical MoE layers with 256, 1024, 4096, 16384, 65536 and 131072 experts. For the hierarchical MoE layers, the ﬁrst level branching factors are 32, 32, 64, 128, 256 and 256, respectively.  

Training: Models are trained on a cluster of 32 Tesla K40 GPUs, except for the last two models, which are trained on clusters of 64 and 128 GPUs so as to have enough memory for all the param- eters. For all models, training batch sizes are approximately 2.5 million words. Models are trained once-through over about 100 billion words.  

We implement several memory optimizations in order to ﬁt up to 1 billion parameters per GPU. First, we do not store the activations of the hidden layers of the experts, but instead recompute them on the backwards pass. Secondly, we modify the optimizer on the expert parameters to require less auxiliary storage:  

The Adam optimizer (Kingma & Ba, 2015) keeps ﬁrst and second moment estimates of the per- parameter gradients. This triples the required memory. To avoid keeping a ﬁrst-moment estimator, we set    $\beta_{1}\,=\,0$  . To reduce the size of the second moment estimator, we replace it with a factored approximation. For a matrix of parameters, instead of maintaining a full matrix of second-moment estimators, we maintain vectors of row-wise and column-wise averages of that matrix. At each step, the matrix of estimators is taken to be the outer product of those two vectors divided by the mean of either one. This technique could similarly be applied to Adagrad (Duchi et al., 2010).  

![Table 8: Model comparison on 100 Billion Word Google News Dataset ](images/23cc74fe34de3599dafd21f3d79243f79072dc7632d23a2ad026851db998f116.jpg)  

Results: We evaluate our model using perplexity on a holdout dataset. Results are reported in Table 8. Perplexity after 100 billion training words is  $39\%$   lower for the 68-billion-parameter MoE model than for the baseline model. It is notable that the measured computational efﬁciency of the largest model (0.30 TFLOPS/GPU) is very low compared to the other models. This is likely a result of the fact that, for purposes of comparison to the other models, we did not increase the training batch size proportionally to the number of GPUs. For comparison, we include results for a computationally matched baseline model consisting of 4 LSTMs, and for an unpruned 5-gram model with Kneser-Ney smoothing (Kneser & Ney, 1995).  

# E M ACHINE  T RANSLATION  - E XPERIMENTAL  D ETAILS  

Model Architecture for Single Language Pair MoE Models: Our model is a modiﬁed version of the GNMT model described in (Wu et al., 2016). To reduce computation, we decrease the number of LSTM layers in the encoder and decoder from 9 and 8 to 3 and 2 respectively. We insert MoE layers in both the encoder (between layers 2 and 3) and the decoder (between layers 1 and 2). We use an attention mechanism between the encoder and decoder, with the ﬁrst decoder LSTM receiving output from and providing input for the attention   5 . All of the layers in our model have input and output dimensionality of 512. Our LSTM layers have 2048 hidden units, with a 512-dimensional output projection. We add residual connections around all LSTM and MoE layers to encourage gradient ﬂow (He et al., 2015). Similar to GNMT, to effectively deal with rare words, we used sub- word units (also known as “wordpieces") (Schuster & Nakajima, 2012) for inputs and outputs in our system.  

We use a shared source and target vocabulary of 32K wordpieces. We also used the same beam search technique as proposed in (Wu et al., 2016).  

We train models with different numbers of experts in the MoE layers. In addition to a baseline model with no MoE layers, we train models with ﬂat MoE layers containing 32 experts, and models with hierarchical MoE layers containing 512 and 2048 experts. The ﬂat MoE layers use  $k=4$   and the hierarchical MoE models use  $k\,=\,2$   at each level of the gating network. Thus, each input is processed by exactly 4 experts in each MoE layer. Each expert in the MoE layer is a feed forward er of size 2048 and ReLU activation. Thus, each expert contains    $[512*$   $2048]+[2048*512]=2M$   ∗  parameters. The output of the MoE layer is passed through a sigmoid function. We use the strictly-balanced gating function described in Appendix F.  

Model Architecture for Multilingual MoE Model: We used the same model architecture as for the single-language-pair models, with the following exceptions: We used noisy-top-k gating as described in Section 2.1, not the scheme from Appendix F. The MoE layers in the encoder and decoder are non-hierarchical MoEs with    $n\,=\,512$   experts, and    $k\,=\,2$  . Each expert has a larger hidden layer of size  8192 . This doubles the amount of computation in the MoE layers, raising the computational budget of the entire model from 85M to 102M ops/timestep.  

Training: We trained our networks using the Adam optimizer (Kingma & Ba, 2015). The base learning rate was increased linearly for the ﬁrst 2000 training steps, held constant for an additional 8000 steps, and decreased after that so as to be proportional to the inverse square root of the step number. For the single-language-pair models, similarly to (Wu et al., 2016), we applied dropout (Zaremba et al., 2014) to the output of all embedding, LSTM and MoE layers, using  $\begin{array}{r}{D r o p P r o b=}\end{array}$  0 . 4 . Training was done synchronously on a cluster of up to 64 GPUs as described in section 3. Each training batch consisted of a set of sentence pairs containing roughly 16000 words per GPU.  

To ensure balanced expert utilization we set    $w_{i m p o r t a n c e}=0.01$   and  $w_{l o a d}=0.01$  , as described in Section 4 and Appendix A.  

Metrics: We evaluated our models using the perplexity and the standard BLEU score metric. We reported tokenized BLEU score as computed by the multi-bleu.pl script, downloaded from the public implementation of Moses (on Github), which was also used in (Luong et al., 2015a).  

Results: Tables 2, 3 and 4 in Section 5.3 show comparisons of our results to other published methods. Figure 4 shows test perplexity as a function of number of words in the (training data’s) source sentences processed for models with different numbers of experts. As can be seen from the Figure, as we increased the number of experts to approach 2048, the test perplexity of our model continued to improve.  

![](images/7bd62bd49ba33e2038817f6979554d50d781d098a94da6196b35aac9a2fe18fb.jpg)  
Figure 4: Perplexity on WMT’  ${14}\;\mathrm{En}{\rightarrow}\;\mathrm{Fr}$   (left) and Google Production   $\mathrm{En}{\rightarrow}\,\mathrm{Fr}$   (right) datasets as a function of number of words processed. The large differences between models at the beginning of training are due to different batch sizes. All models incur the same computational budget (85M ops/timestep) except the one with no experts.  

We found that the experts indeed become highly specialized by syntax and/or semantics, as can be seen in Table 9. For example, one expert is used when the indeﬁnite article “a" introduces the direct object in a verb phrase indicating importance or leadership.  

Table 9: Contexts corresponding to a few of the 2048 experts in the MoE layer in the encoder portion of the WMT’  ${14}\;\mathrm{En}{\rightarrow}\;\mathrm{Fr}$  slation model. For each expert    $i$  , we sort the inputs in a training batch in decreasing order of  $G(x)_{i}$  , and show the words surrounding the corresponding positions in the input sentences.  

![](images/6bdfc4fae9364423cf32eeb325fcabe9c2bbe90fd13deca53ea354fa60af428a.jpg)  

# F S TRICTLY  B ALANCED  G ATING  

Due to some peculiarities in our infrastructure which have since been ﬁxed, at the time we ran some of the machine translation experiments, our models ran faster if every expert received exactly the same batch size. To accommodate this, we used a different gating function which we describe below.  

Recall that we deﬁne the softmax gating function to be:  

$$
G_{\sigma}(x)=S o f t m a x(x\cdot W_{g})
$$  

Sparse Gating (alternate formulation): To obtain a sparse gating vector, we multiply    $G_{\sigma}(x)$  component-wise with a sparse mask    $M(G_{\sigma}(x))$   and normalize the output. The mask itself is a function of  $G_{\sigma}(x)$   and speciﬁes which experts are assigned to each input example:  

$$
G(x)_{i}=\frac{G_{\sigma}(x)_{i}M(G_{\sigma}(x))_{i}}{\sum_{j=1}^{n}G_{\sigma}(x)_{j}M(G_{\sigma}(x))_{j}}
$$  

Top-K Mask: To implement top-  $\cdot\mathrm{k}$   gating in this formulation, we would let  $M(v)=T o p K(v,k)$  , where:  

$$
T o p K(v,k)_{i}={\left\{\begin{array}{l l}{1}&{{\mathrm{if~}}v_{i}{\mathrm{~is~in~the~top~}}k{\mathrm{~elements~of~}}v.}\\ {0}&{{\mathrm{otherwise.}}}\end{array}\right.}
$$  

Batchwise Mask: To force each expert to receive the exact same number of examples, we intro- duce an alternative mask function,    $\bar{M_{b a t c h w i s e}}(X,m)$  , which operates over batches of input vectors. Instead of keeping the top  $k$   values per example, we keep the top    $m$   values per expert across the training batch, where    $\textstyle m={\frac{k|X|}{n}}$  , so that each example is sent to an average of    $k$   experts.  

$$
M_{b a t c h w i s e}(X,m)_{j,i}={\left\{\begin{array}{l l}{1}&{{\mathrm{if}}\ X_{j,i}\ {\mathrm{is~in~the~top}}\ m\ {\mathrm{values~for~to~expert~}}i}\\ {0}&{{\mathrm{otherwise}}}\end{array}\right.}
$$  

As our experiments suggest and also observed in (Ioffe & Szegedy, 2015), using a batchwise func- tion during training (such as    $M_{b a t c h w i s e})$  ) requires modiﬁcations to the inference when we may not have a large batch of examples. Our solution to this is to train a vector    $T$   of per-expert threshold values to approximate the effects of the batchwise mask. We use the following mask at inference time:  

$$
M_{t h r e s h o l d}(x,T)_{i}={\left\{\begin{array}{l l}{1}&{{\mathrm{if}}\ x_{i}>T_{i}}\\ {0}&{{\mathrm{otherwise}}}\end{array}\right.}
$$  

To learn the threshold values, we apply an additional loss at training time which is minimized when the batchwise mask and the threshold mask are identical.  

$$
L_{b a t c h w i s e}(X,T,m)=\sum_{j=1}^{|X|}\sum_{i=1}^{n}(M_{t h r e s h o l d}(x,T)_{i}-M_{b a t c h w i s e}(X,m)_{j,i})(X_{j,i}-T_{i})
$$  

G A TTENTION  F UNCTION  

The attention mechanism described in GNMT (Wu et al., 2016) involves a learned “Attention Func- tion"    $A(x_{i},y_{j})$   which takes a “source vector"  $x_{i}$   and a “target vector"  $y_{j}$  , and must be computed for every source time step    $i$   and target time step    $j$  . In GNMT, the attention function is implemented as a feed forward neural network with a hidden layer of size    $n$  . It can be expressed as:  

$$
A_{G N M T}(x_{i},y_{j})=\sum_{d=1}^{n}V_{d}t a n h((x_{i}U)_{d}+(y_{j}W)_{d})
$$  

Where    $U$   and  $W$   are trainable weight matrices and  $V$   is a trainable weight vector.  

For performance reasons, in our models, we used a slightly different attention function:  

$$
A(x_{i},y_{j})=\sum_{d=1}^{n}V_{d}t a n h((x_{i}U)_{d})t a n h((y_{j}W)_{d})
$$  

With our attention function, we can simultaneously compute the attention function on multiple source time steps and multiple target time steps using optimized matrix multiplications. We found little difference in quality between the two functions.  