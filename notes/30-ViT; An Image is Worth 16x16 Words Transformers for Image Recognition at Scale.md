# A N  I MAGE IS  W ORTH  16 X 16 W ORDS : T RANSFORMERS FOR  I MAGE  R ECOGNITION AT  S CALE  

# Alexey Dosovitskiy ∗ , † , Lucas Beyer ∗ , Alexander Kolesnikov ∗ , Dirk Weissenborn ∗ , Xiaohua Zhai ∗ , Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby ∗ , †  

∗ equal technical contribution,  † equal advising Google Research, Brain Team { adosovitskiy, neilhoulsby } @google.com  

# A BSTRACT  

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classiﬁcation tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring sub- stantially fewer computational resources to train.  

# 1 I NTRODUCTION  

Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then ﬁne-tune on a smaller task-speciﬁc dataset (Devlin et al., 2019). Thanks to Transformers’ computational efﬁciency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020). With the models and datasets growing, there is still no sign of saturating performance.  

In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while theoretically efﬁcient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNet- like architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020).  

Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modiﬁcations. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Trans- former. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classiﬁcation in supervised fashion.  

When trained on mid-sized datasets such as ImageNet without strong regularization, these mod- els yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufﬁcient amounts of data.  

However, the picture changes if the models are trained on larger datasets (14M-300M images). We ﬁnd that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufﬁcient scale and transferred to tasks with fewer datapoints. When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of  $88.55\%$   on ImageNet,  $90.72\%$   on ImageNet-ReaL,  $94.55\%$   on CIFAR-100, and    $77.63\%$   on the VTAB suite of 19 tasks.  

# 2 R ELATED  W ORK  

Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since be- come the state of the art method in many NLP tasks. Large Transformer-based models are often pre-trained on large corpora and then ﬁne-tuned for the task at hand: BERT (Devlin et al., 2019) uses a denoising self-supervised pre-training task, while the GPT line of work uses language mod- eling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).  

Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past. Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020). In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global self- attention in order to be applicable to images. An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a). Many of these specialized attention architectures demonstrate promising results on computer vision tasks, but require complex engineering to be implemented efﬁciently on hardware accelerators.  

Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size    $2\times2$  from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020) use a small patch size of    $2\times2$   pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.  

There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms of self-attention, e.g. by augmenting feature maps for image classiﬁcation (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classiﬁcation (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or uniﬁed text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019).  

Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers to image pixels after reducing image resolution and color space. The model is trained in an unsu- pervised fashion as a generative model, and the resulting representation can then be ﬁne-tuned or probed linearly for classiﬁcation performance, achieving a maximal accuracy of  $72\%$   on ImageNet.  

Our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. The use of additional data sources allows to achieve state-of- the-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020). Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from large scale datasets such as ImageNet-21k and JFT-300M. We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models used in prior works.  

![](images/02bb550131ab7c9273dda634d4e60942ae40b547091d45d08178d9c4ec41551d.jpg)  
Figure 1: Model overview. We split an image into ﬁxed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classiﬁcation, we use the standard approach of adding an extra learnable “classiﬁcation token” to the sequence. The illustration of the Transformer encoder was inspired by Vaswani et al. (2017).  

# 3 M ETHOD  

In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efﬁcient implementations – can be used almost out of the box.  

# 3.1 V ISION  T RANSFORMER  (V I T)  

An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image  $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$    into a sequen  of ﬂattened 2D patches  $\mathbf{x}_{p}\in\mathbb{R}^{N\times(P^{2}\cdot C)}$  , where    $(H,W)$   is the resolution image,  C  is the number of channels,  $(P,P)$   is the resolution of each image patch, and  $N=H W/P^{2}$  is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size    $D$   through all of its layers, so we ﬂatten the patches and map to    $D$   dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.  

Similar to BERT’s  [class]  token, we prepend a learnable embedding to the sequence of embed- ded patches   $(\mathbf{z}_{0}^{0}=\mathbf{x}_{\mathrm{class}})$  ), whose state at the output of the Transformer encoder   $\bar{(\mathbf{z}_{L}^{0})}$  ) serves as the image representation  y  (Eq. 4). Both during pre-training and ﬁne-tuning, a classiﬁcation head is at- tached to    $\mathbf{z}_{L}^{0}$  . The classiﬁcation head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at ﬁne-tuning time.  

Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed signiﬁcant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder.  

The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self- attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).  

The MLP contains two layers with a GELU non-linearity.  

$$
\begin{array}{r l r l}&{\mathbf{z}_{0}=[\mathbf{x}_{\mathrm{class}};\,\mathbf{x}_{p}^{1}\mathbf{E};\,\mathbf{x}_{p}^{2}\mathbf{E};\,\cdots\,;\,\mathbf{x}_{p}^{N}\mathbf{E}]+\mathbf{E}_{p o s},}&&{\mathbf{E}\in\mathbb{R}^{(P^{2}\cdot C)\times D},\,\mathbf{E}_{p o s}\in\mathbb{R}^{(N+1)\times D}}\\ &{\mathbf{z}^{\prime}{}_{\ell}=\mathrm{MSA}\big(\mathrm{LN}(\mathbf{z}_{\ell-1})\big)+\mathbf{z}_{\ell-1},}&&{\ell=1\ldots L}\\ &{\mathbf{z}_{\ell}=\mathrm{MLP}\big(\mathrm{LN}(\mathbf{z}^{\prime}{}_{\ell})\big)+\mathbf{z}^{\prime}{}_{\ell},}&&{\ell=1\ldots L}\\ &{\mathbf{y}=\mathrm{LN}(\mathbf{z}_{L}^{0})}\end{array}
$$  

Inductive bias. We note that Vision Transformer has much less image-speciﬁc inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and transla- tionally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at ﬁne-tuning time for adjusting the position embeddings for images of different resolution (as de- scribed below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.  

Hybrid Architecture. As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection  $\mathbf{E}$   (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply ﬂattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classiﬁcation input embedding and position embeddings are added as described above.  

# 3.2 F INE - TUNING AND  H IGHER  R ESOLUTION  

Typically, we pre-train ViT on large datasets, and ﬁne-tune to (smaller) downstream tasks. For this, we rem  the pre-trained prediction head and attach a zero-initialized    $D\times K$   feedforward layer, where  K  is the number of downstream classes. It is often beneﬁcial to ﬁne-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.  

# 4 E XPERIMENTS  

We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show that self-supervised ViT holds promise for the future.  

4.1 S ETUP  

Datasets.  To explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with   $18\mathbf{k}$   classes and 303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008). For these datasets, pre-processing follows Kolesnikov et al. (2020).  

![Table 1: Details of Vision Transformer model variants. ](images/4f8bad417e3db3e50a6d9babf86cfb980f8a6509c0b7495950e17947d2da3d69.jpg)  

We also evaluate on the 19-task VTAB classiﬁcation suite (Zhai et al., 2019b). VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into three groups:  Natural  – tasks like the above, Pets, CIFAR, etc.  Specialized  – medical and satellite imagery, and  Structured  – tasks that require geometric understanding like localization.  

Model Variants.  We base ViT conﬁgurations on those used for BERT (Devlin et al., 2019), as summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we add the larger “Huge” model. In what follows we use brief notation to indicate the model size and the input patch size: for instance, ViT-L/16 means the “Large” variant with    $16\times16$   input patch size. Note that the Transformer’s sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive.  

For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization lay- ers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized convolutions (Qiao et al., 2019). These modiﬁcations improve transfer (Kolesnikov et al., 2020), and we denote the modiﬁed model “ResNet (BiT)”. For the hybrids, we feed the intermediate fea- ture maps into ViT with patch size of one “pixel”. To experiment with different sequence lengths, we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same number of layers in stage 3 (keeping the total number of layers), and take the output of this extended stage 3. Option (ii) results in a   $4\mathrm{{x}}$   longer sequence length, and a more expensive ViT model.  

Training & Fine-tuning.  We train all models, including ResNets, using Adam (Kingma & Ba, 2015) with  $\beta_{1}=0.9$  ,    $\beta_{2}=0.999$  , a batch size of 4096 and apply a high weight decay of  0 . 1 , which we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning rate warmup and decay, see Appendix B.1 for details. For ﬁne-tuning we use SGD with momentum, batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we ﬁne-tuned at higher resolution:  512  for ViT-L/16 and  518  for ViT-H/14, and also used Polyak & Juditsky (1992) averaging with a factor of  0 . 9999  (Ramachandran et al., 2019; Wang et al., 2020b).  

Metrics.  We report results on downstream datasets either through few-shot or ﬁne-tuning accuracy. Fine-tuning accuracies capture the performance of each model after ﬁne-tuning it on the respective dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to  $\{-\bar{1},1\}^{K}$    target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on ﬁne-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-ﬂy evaluation where ﬁne-tuning would be too costly.  

# 4.2 C OMPARISON TO  S TATE OF THE  A RT  

We ﬁrst compare our largest models – ViT-H/14 and ViT-L/16 – to state-of-the-art CNNs from the literature. The ﬁrst comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which performs supervised transfer learning with large ResNets. The second is Noisy Student (Xie et al., 2020), which is a large EfﬁcientNet trained using semi-supervised learning on ImageNet and JFT- 300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU v3 cores (2 per chip) used for training multiplied by the training time in days.  

Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L (which is pre-trained on the same dataset) on all tasks, while requiring substantially less computa- tional resources to train. The larger model, ViT-H/14, further improves the performance, especially on the more challenging datasets – ImageNet, CIFAR-100, and the VTAB suite. Interestingly, this  

![](images/8bb41d374b2a281fe81c5ce29a5704ea749dc22b7b8aca78410641f3f4ef64d0.jpg)  

Table 2: Comparison with state of the art on popular image classiﬁcation benchmarks. We re- port mean and standard deviation of the accuracies, averaged over three ﬁne-tuning runs. Vision Transformer models pre-trained on the JFT-300M dataset outperform ResNet-based baselines on all datasets, while taking substantially less computational resources to pre-train. ViT pre-trained on the smaller public ImageNet-21k dataset performs well too.   ∗ Slightly improved    $88.5\%$   result reported in Touvron et al. (2020).  

![](images/9d1bf588255fdc6f0dc85bd4518e7b10d03a3d1b19f65646976bd6645738a4de.jpg)  
Figure 2: Breakdown of VTAB performance in  Natural ,  Specialized , and  Structured  task groups.  

model still took substantially less compute to pre-train than prior state of the art. However, we note that pre-training efﬁciency may be affected not only by the architecture choice, but also other pa- rameters, such as training schedule, optimizer, weight decay, etc. We provide a controlled study of performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in ap- proximately 30 days.  

Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA methods on this benchmark: BiT, VIVI – a ResNet co-trained on ImageNet and Youtube (Tschannen et al., 2020), and S4L – supervised plus semi-supervised learning on ImageNet (Zhai et al., 2019a). ViT-H/14 outperforms BiT-R152x4, and other methods, on the  Natural  and  Structured  tasks. On the Specialized  the performance of the top two models is similar.  

# 4.3 P RE - TRAINING  D ATA  R EQUIREMENTS  

The Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of experiments.  

First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT- 300M. To boost the performance on the smaller datasets, we optimize three basic regularization parameters – weight decay, dropout, and label smoothing. Figure 3 shows the results after ﬁne- tuning to ImageNet (results on other datasets are shown in Table 5) 2 . When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full beneﬁt of larger models. Figure 3 also shows the performance  

![](images/70d1bfa8dd6a39dc0dc1df1f40db51d5fe1444f0b34e3b49ce042d173a31f70d.jpg)  
Figure 3: Transfer to ImageNet. While Figure 4: Linear few-shot evaluation on Ima- large ViT models perform worse than BiT geNet versus pre-training size. ResNets per- ResNets (shaded area) when pre-trained on form better with smaller pre-training datasets small datasets, they shine when pre-trained on but plateau sooner than ViT, which performs larger datasets. Similarly, larger ViT variants better with larger pre-training. ViT-b is ViT-B overtake smaller ones as the dataset grows. with all hidden dimensions halved.  

![](images/d8156faff3cd8eecf92f00f767672d5402687f5a42cce80a492847ca042ff588.jpg)  
Total pre-training compute [exaFLOPs]  

region spanned by BiT models of different sizes. The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes.  

Second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT- 300M dataset. We do not perform additional regularization on the smaller subsets and use the same hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the effect of regularization. We do, however, use early-stopping, and report the best validation accuracy achieved during training. To save compute, we report few-shot linear accuracy instead of full ﬁne- tuning accuracy. Figure 4 contains the results. Vision Transformers overﬁt more than ResNets with comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than ResNet50; it performs much worse on the 9M subset, but better on  $90\mathrm{M}+$   subsets. The same is true for ResNet152x2 and ViT-L/16. This result reinforces the intuition that the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufﬁcient, even beneﬁcial.  

Overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB (Table 2) seem promising for very low-data transfer. Further analysis of few-shot properties of ViT is an exciting direction of future work.  

# 4.4 S CALING  S TUDY  

We perform a controlled scaling study of different models by evaluating transfer performance from JFT-300M. In this setting data size does not bottleneck the models’ performances, and we assess performance versus pre-training cost of each model. The model set includes: 7 ResNets, R50x1, R50x2 R101x1,   ${\bf R}152x1$  ,   $\mathbf{R}152\mathbf{x}2$  , pre-trained for 7 epochs, plus   $\mathbf{R}152\mathbf{x}2$   and   ${\bf R}200{\bf x}3$   pre-trained for 14 epochs; 6 Vision Transformers, ViT-B/32, B/16, L/32, L/16, pre-trained for 7 epochs, plus  $\mathrm{L}/16$   and H/14 pre-trained for 14 epochs; and 5 hybrids, R50+ViT-B/32, B/16, L/32, L/16 pre- trained for 7 epochs, plus   $\mathrm{R50+ViT-L/16}$  pre-trained for 14 epochs (for hybrids, the number at the end of the model name stands not for the patch size, but for the total dowsampling ratio in the ResNet backbone).  

Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.5 for details on computational costs). Detailed results per model are provided in Table 6 in the Ap- pendix. A few patterns can be observed. First, Vision Transformers dominate ResNets on the performance/compute trade-off. ViT uses approximately    $2\mathrm{~-~}4\times$   less compute to attain the same performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small compu- tational budgets, but the difference vanishes for larger models. This result is somewhat surprising, since one might expect convolutional local feature processing to assist ViT at any size. Third, Vision Transformers appear not to saturate within the range tried, motivating future scaling efforts.  

# 4.5 I NSPECTING  V ISION  T RANSFORMER  

To begin to understand how the Vision Transformer processes im- age data, we analyze its internal representations. The ﬁrst layer of the Vision Transformer linearly projects the ﬂattened patches into a lower-dimensional space (Eq. 1). Figure 7 (left) shows the top prin- cipal components of the the learned embedding ﬁlters. The com- ponents resemble plausible basis functions for a low-dimensional representation of the ﬁne structure within each patch.  

After the projection, a learned position embedding is added to the patch representations. Figure 7 (center) shows that the model learns to encode distance within the image in the similarity of position em- beddings, i.e. closer patches tend to have more similar position em- beddings. Further, the row-column structure appears; patches in the same row/column have similar embeddings. Finally, a sinusoidal structure is sometimes apparent for larger grids (Appendix D). That the position embeddings learn to represent 2D image topology ex- plains why hand-crafted 2D-aware embedding variants do not yield improvements (Appendix D.4).  

![](images/c29c2db4785850209bb0177f01abaae52d1fe9c66ab788b63631b0111e653980.jpg)  

Self-attention allows ViT to integrate information across the entire Figure 6: Representative ex- image even in the lowest layers. We investigate to what degree amples of attention from the the network makes use of this capability. Speciﬁcally, we compute output token to the input the average distance in image space across which information is space. See Appendix D.7 for integrated, based on the attention weights (Figure 7, right). This details. “attention distance” is analogous to receptive ﬁeld size in CNNs.  

We ﬁnd that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model. Other attention heads have consistently small attention distances in the low layers. This highly localized attention is less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right), suggesting that it may serve a similar function as early convolutional layers in CNNs. Further, the attention distance increases with network depth. Globally, we ﬁnd that the model attends to image regions that are semantically relevant for classiﬁcation (Figure 6).  

# 4.6 S ELF - SUPERVISION  

Transformers show impressive performance on NLP tasks. However, much of their success stems not only from their excellent scalability but also from large scale self-supervised pre-training (Devlin  

![](images/419c43429eef50f3f00d5792a6e64803e6fa1fb0ed8eb6fdbb6f2ba203d41eba.jpg)  
Figure 7:  Left:  Filters of the initial linear embedding of RGB values of ViT-L/32.  Center:  Sim- ilarity of position embeddings of ViT-L/32. Tiles show the cosine similarity between the position embedding of the patch with the indicated row and column and the position embeddings of all other patches.  Right:  Size of attended area by head and network depth. Each dot shows the mean attention distance across images for one of 16 heads at one layer. See Appendix D.7 for details.  

et al., 2019; Radford et al., 2018). We also perform a preliminary exploration on  masked patch prediction  for self-supervision, mimicking the masked language modeling task used in BERT. With self-supervised pre-training, our smaller ViT-B/16 model achieves   $79.9\%$   accuracy on ImageNet, a signiﬁcant improvement of  $2\%$   to training from scratch, but still  $4\%$   behind supervised pre-training. Appendix B.1.2 contains further details. We leave exploration of contrastive pre-training (Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; H´ enaff et al., 2020) to future work.  

# 5 C ONCLUSION  

We have explored the direct application of Transformers to image recognition. Unlike prior works using self-attention in computer vision, we do not introduce image-speciﬁc inductive biases into the architecture apart from the initial patch extraction step. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP. This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets. Thus, Vision Transformer matches or exceeds the state of the art on many image classiﬁcation datasets, whilst being relatively cheap to pre-train.  

While these initial results are encouraging, many challenges remain. One is to apply ViT to other computer vision tasks, such as detection and segmentation. Our results, coupled with those in Carion et al. (2020), indicate the promise of this approach. Another challenge is to continue exploring self- supervised pre-training methods. Our initial experiments show improvement from self-supervised pre-training, but there is still large gap between self-supervised and large-scale supervised pre- training. Finally, further scaling of ViT would likely lead to improved performance.  

# A CKNOWLEDGEMENTS  

The work was performed in Berlin, Z¨ urich, and Amsterdam. We thank many colleagues at Google for their help, in particular Andreas Steiner for crucial help with the infrastructure and the open- source release of the code; Joan Puigcerver and Maxim Neumann for help with the large-scale training infrastructure; Dmitry Lepikhin, Aravindh Mahendran, Daniel Keysers, Mario Luˇ ci´ c, Noam Shazeer, Ashish Vaswani, and Colin Raffel for useful discussions.  

# R EFERENCES  

Samira Abnar and Willem Zuidema. Quantifying attention ﬂow in transformers. In  ACL , 2020.  

Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In  NeurIPS , 2019.  

Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In ICLR , 2019. I. Bello, B. Zoph, Q. Le, A. Vaswani, and J. Shlens. Attention augmented convolutional networks. In  ICCV , 2019. Lucas Beyer, Olivier J. H´ enaff, Alexander Kolesnikov, Xiaohua Zhai, and A¨ aron van den Oord. Are we done with imagenet?  arXiv , 2020. Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.  arXiv , 2020. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In  ECCV , 2020. Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. Generative pretraining from pixels. In  ICML , 2020a. Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In  ICML , 2020b. Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: UNiversal Image-TExt Representation Learning. In  ECCV , 2020c. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers.  arXiv , 2019. Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self- attention and convolutional layers. In  ICLR , 2020. J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In  CVPR , 2009. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In  NAACL , 2019. Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D’Amour, Dan Moldovan, Sylvan Gelly, Neil Houlsby, Xiaohua Zhai, and Mario Lucic. On robustness and transferability of convo- lutional neural networks.  arXiv , 2020. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog- nition. In  CVPR , 2016. Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In  CVPR , 2020. Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. Axial attention in multidi- mensional transformers.  arXiv , 2019. Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In  CVPR , 2018. Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In  ICCV , 2019. Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. In  ICCV , 2020. Olivier J. H´ enaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aaron van den Oord. Data-efﬁcient image recognition with contrastive predictive coding. In ICML , 2020.  

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 2015. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In  ICLR , 2015. Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (BiT): General visual representation learning. In  ECCV , 2020. Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classiﬁcation with deep convo- lutional neural networks. In  NIPS , 2012. Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel. Backpropa- gation applied to handwritten zip code recognition.  Neural Computation , 1:541–551, 1989. Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding.  arXiv , 2020. Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A Simple and Performant Baseline for Vision and Language. In  Arxiv , 2019. Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot atten- tion.  arXiv , 2020. Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visi- olinguistic Representations for Vision-and-Language Tasks. In  NeurIPS . 2019. Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In  ECCV , 2018. M. Nilsback and A. Zisserman. Automated ﬂower classiﬁcation over a large number of classes. In ICVGIP , 2008. Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In  CVPR , 2012. Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In  ICML , 2018. B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging.  SIAM Journal on Control and Optimization , 30(4):838–855, 1992. doi: 10.1137/0330046. URL https://doi.org/10.1137/0330046 . Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. Weight standardization.  arXiv preprint arXiv:1903.10520 , 2019. Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language under- standing with unsupervised learning.  Technical Report , 2018. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners.  Technical Report , 2019. Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. In  NeurIPS , 2019. Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable ef- fectiveness of data in deep learning era. In  ICCV , 2017. Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In  ICCV , 2019.  

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy. In  NeurIPS . 2019. Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy: Fixefﬁcientnet.  arXiv preprint arXiv:2003.08237 , 2020. Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. Self-supervised learning of video-induced visual invariances. In  Pro- ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , June 2020. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In  NIPS , 2017. Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. In  ECCV , 2020a. Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. arXiv preprint arXiv:2003.07853 , 2020b. Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In  ACL , 2019. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In CVPR , 2018. Dirk Weissenborn, Oscar T¨ ackstr¨ om, and Jakob Uszkoreit. Scaling autoregressive video models. In ICLR , 2019. Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based image representation and processing for computer vision.  arxiv , 2020. Yuxin Wu and Kaiming He. Group normalization. In  ECCV , 2018. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classiﬁcation. In  CVPR , 2020. Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer.   $\mathrm{S}^{4}\mathrm{L}$  : Self-Supervised Semi- Supervised Learning. In  ICCV , 2019a. Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark.  arXiv preprint arXiv:1910.04867 , 2019b. Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In CVPR , 2020.  

![](images/2f3256d696433941e98eeea6f61eb267769cae54d5fa4b1d9cad5b4b237de7d3.jpg)  
Table 3: Hyperparameters for training. All models are trained with a batch size of 4096 and learn- ing rate warmup of   $10\mathbf{k}$   steps. For ImageNet we found it beneﬁcial to additionally apply gradient clipping at global norm 1. Training resolution is 224.  

# A PPENDIX  

# A M ULTIHEAD  S ELF - ATTENTION  

Standard  qkv  self-attention (SA, Vaswani et al. (2017)) is a popular building block for neural archi- tecture  For each element in an input sequence  $\mathbf{z}\in\dot{\mathbb{R}}^{N\times\hat{D}}$  , we compute a weighted sum over all values  v  in the sequence. The attention weights    $A_{i j}$   are based on the pairwise similarity between two elements of the sequence and their respective query  $\mathbf{q}^{i}$    and key  $\mathbf{k}^{j}$    representations.  

$$
\begin{array}{r l r}{[\mathbf{q},\mathbf{k},\mathbf{v}]=\mathbf{z}\mathbf{U}_{q k v}\quad}&{\quad}&{\mathbf{U}_{q k v}\in\mathbb{R}^{D\times3D_{h}},}\\ {A=\operatorname{softmax}\left(\mathbf{q}\mathbf{k}^{\top}/\sqrt{D_{h}}\right)\quad}&{\quad}&{\quad A\in\mathbb{R}^{N\times N},}\\ {\mathrm{S}\mathrm{A}(\mathbf{z})=A\mathbf{v}\,.}\end{array}
$$  

Multihead self-attention (MSA) is an extension of SA in which we run    $k$   self-attention operations, called “heads”, in parallel, and project their concatenated outputs. To keep compute and number of parameters constant when changing    $k$  ,  $D_{h}$   (Eq. 5) is typically set to    $D/k$  .  

$$
\begin{array}{r}{\operatorname{MSA}(\mathbf{z})=\operatorname{[SA}_{1}(z);\operatorname{SA}_{2}(z);\cdot\cdot\cdot;\operatorname{SA}_{k}(z)]\,\mathbf{U}_{m s a}\qquad\qquad\mathbf{U}_{m s a}\in\mathbb{R}^{k\cdot D_{h}\times D_{s}}}\end{array}
$$  

# B E XPERIMENT DETAILS  

B.1 T RAINING  

Table 3 summarizes our training setups for our different models. We found strong regularization to be key when training models from scratch on ImageNet. Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings. Hybrid models are trained with the exact setup as their ViT counterparts. Finally, all training is done on resolution 224.  

# B.1.1 F INE - TUNING  

We ﬁne-tune all ViT models using SGD with a momentum of 0.9. We run a small grid search over learning rates, see learning rate ranges in Table 4. To do so, we use small sub-splits from the training set (  $10\%$   for Pets and Flowers,  $2\%$   for CIFAR,  $1\%$   ImageNet) as development set and train on the remaining data. For ﬁnal results we train on the entire training set and evaluate on the respective test data. For ﬁne-tuning ResNets and hybrid models we use the exact same setup, with the only exception of ImageNet where we add another value  0 . 06  to the learning rate sweep. Additionally,  

![Table 4: Hyperparameters for ﬁne-tuning. All models are ﬁne-tuned with cosine learning rate decay, a batch size of 512, no weight decay, and grad clipping at global norm 1. If not mentioned otherwise, ﬁne-tuning resolution is 384. ](images/2435d034b672f9e12ddacb4271fd04615ce70709294a34a670260c11b8a0e6c0.jpg)  

for ResNets we also run the setup of Kolesnikov et al. (2020) and select the best results across this run and our sweep. Finally, if not mentioned otherwise, all ﬁne-tuning experiments run at 384 resolution (running ﬁne-tuning at different resolution than training is common practice (Kolesnikov et al., 2020)).  

When transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset. We found this to be a little more robust than simply re-initializing the very last layer.  

For VTAB we follow the protocol in Kolesnikov et al. (2020), and use the same hyperparameter setting for all tasks. We use a learning rate of  0 . 01  and train for  2500  steps (Tab. 4). We chose this setting by running a small sweep over two learning rates and two schedules, and selecting the setting with the highest VTAB score on the 200-example validation sets. We follow the pre-processing used in Kolesnikov et al. (2020), except that we do not use task-speciﬁc input resolutions. Instead we ﬁnd that Vision Transformer beneﬁts most from a high resolution   $(384\times384)$   for all tasks.  

# B.1.2SELF-SUPERVISION  

We employ the  masked patch prediction  objective for preliminary self-supervision experiments. To do so we corrupt   $50\%$   of patch embeddings by either replacing their embeddings with a learnable

 [mask]  embedding   $(80\%)$  , a random other patch embedding   $(10\%)$   or just keeping them as is

  $(10\%)$  . This setup is very similar to the one used for language by Devlin et al. (2019). Finally, we predict the 3-bit, mean color (i.e., 512 colors in total) of every corrupted patch using their respective patch representations.  

We trained our self-supervised model for 1M steps (ca. 14 epochs) with batch size 4096 on JFT. We use Adam, with a base learning rate of    $2\cdot10^{-4}$  , warmup of   $10\mathbf{k}$   steps and cosine learning rate decay. As prediction targets for pretraining we tried the following settings: 1) predicting only the mean, 3bit color (i.e., 1 prediction of 512 colors), 2) predicting a  $4\times4$   downsized version of the  $16\times16$  patch with 3bit colors in parallel (i.e., 16 predictions of 512 colors), 3) regression on the full patch using L2 (i.e., 256 regressions on the 3 RGB channels). Surprisingly, we found that all worked quite well, though L2 was slightly worse. We report ﬁnal results only for option 1) because it has shown best few-shot performance. We also experimented with   $15\%$   corruption rate as used by Devlin et al. (2019) but results were also slightly worse on our few-shot metrics.  

Lastly, we would like to remark that our instantiation of masked patch prediction doesn’t require such an enormous amount of pretraining nor a large dataset such as JFT in order to lead to sim- ilar performance gains on ImageNet classiﬁcation. That is, we observed diminishing returns on downstream performance after   $100\mathrm{k}$   pretraining steps, and see similar gains when pretraining on ImageNet.  

# C A DDITIONAL  R ESULTS  

We report detailed results corresponding to the ﬁgures presented in the paper. Table 5 corresponds to Figure 3 from the paper and shows transfer performance of different ViT models pre-trained on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. Table 6 corresponds to  

![](images/e8cbd447941dc497617f9f52e9c159ae09ffa7c0251f3dcbe9c9c01fad3007c9.jpg)  
Table 5: Top1 accuracy (in  $\%$  ) of Vision Transformer on various datasets when pre-trained on Im- ageNet, ImageNet-21k or JFT300M. These values correspond to Figure 3 in the main text. Models are ﬁne-tuned at 384 resolution. Note that the ImageNet results are computed without additional techniques (Polyak averaging and 512 resolution images) used to achieve results in Table 2.  

![](images/c193112cfe7f93c3a91038064543420c6cfa7aea790725c86d0074114d4a0ca8.jpg)  
Table 6: Detailed results of model scaling experiments. These correspond to Figure 5 in the main paper. We show transfer accuracy on several datasets, as well as the pre-training compute (in ex- aFLOPs).  

Figure 5 from the paper and shows the transfer performance of ViT, ResNet, and hybrid models of varying size, as well as the estimated computational cost of their pre-training.  

# D A DDITIONAL  A NALYSES  

# D.1SGD VS. ADAM FOR RESNETS  

ResNets are typically trained with SGD and our use of Adam as optimizer is quite unconventional. Here we show the experiments that motivated this choice. Namely, we compare the ﬁne-tuning  

![](images/35a3856636ed525ef2f0ea6b1ac520be7c8ca76539d5e8e02e87bdb83a9e85de.jpg)  

![](images/aadae6a8da97416b1fd88f166a8b281539fe54298f8039c1e4b263682ebf9553.jpg)  
Table 7: Fine-tuning ResNet models pre-trained with Adam and SGD.  

performance of two ResNets – 50x1 and   $152\mathrm{x}2$   – pre-trained on JFT with SGD and Adam. For SGD, we use the hyperparameters recommended by Kolesnikov et al. (2020). Results are presented in Table 7. Adam pre-training outperforms SGD pre-training on most datasets and on average. This justiﬁes the choice of Adam as the optimizer used to pre-train ResNets on JFT. Note that the absolute numbers are lower than those reported by Kolesnikov et al. (2020), since we pre-train only for  7  epochs, not  30 .  

# D.2 T RANSFORMER SHAPE  

We ran ablations on scaling different dimensions of the Transformer architecture to ﬁnd out which are best suited for scaling to very large models. Figure 8 shows 5-shot performance on ImageNet for different conﬁgurations. All conﬁgurations are based on a ViT model with  8  layers,    $D=1024$  ,  $D_{M L P}\,=\,2048$   and a patch size of  32 , the intersection of all lines. We can see that scaling the depth results in the biggest improvements which are clearly visible up until 64 layers. However, diminishing returns are already visible after 16 layers. Interestingly, scaling the width of the net- work seems to result in the smallest changes. Decreasing the patch size and thus increasing the effective sequence length shows surprisingly robust improvements without introducing parameters. These ﬁndings suggest that compute might be a better predictor of performance than the number of parameters, and that scaling should emphasize depth over width if any. Overall, we ﬁnd that scaling all dimensions proportionally results in robust improvements.  

# D.3 H EAD  T YPE AND  CLASS  TOKEN  

In order to stay as close as possible to the original Transformer model, we made use of an additional [class]  token, which is taken as image representation. The output of this token is then trans- formed into a class prediction via a small multi-layer perceptron (MLP) with  tanh  as non-linearity in the single hidden layer.  

This design is inherited from the Transformer model for text, and we use it throughout the main paper. An initial attempt at using only image-patch embeddings, globally average-pooling (GAP) them, followed by a linear classiﬁer—just like ResNet’s ﬁnal feature map—performed very poorly. However, we found that this is neither due to the extra token, nor to the GAP operation. Instead,  

![](images/60f4a06a227d8974940a01c364f3519fadd6b3e5d9a519a55acb6a843205e29a.jpg)  

![Figure 9: Comparison of class-token and global average pooling classiﬁers. Both work similarly well, but require different learning-rates. ](images/3626702e53a802da2968a6904aa8bc5370dade82fa400e4c9993b5bb6d10d4e7.jpg)  

Table 8: Results of the ablation study on positional embeddings with ViT-B/16 model evaluated on ImageNet 5-shot linear.  

the difference in performance is fully explained by the requirement for a different learning-rate, see Figure 9.  

# D.4POSITIONAL EMBEDDING  

We ran ablations on different ways of encoding spatial information using positional embedding. We tried the following cases:  

• Providing no positional information: Considering the inputs as a  bag of patches . • 1-dimensional positional embedding: Considering the inputs as a sequence of patches in the raster order (default across all other experiments in this paper). • 2-dimensional positional embedding: Considering the inputs as a grid of patches in two dimensions. In this case, two sets of embeddings are learned, each for one of the axes,  $X$  -embedding, and    $Y$  -embedding, each with size    $D/2$  . Then, based on the coordinate on the path in the input, we concatenate the    $X$   and    $Y$   embedding to get the ﬁnal positional embedding for that patch. • Relative positional embeddings: Considering the relative distance between patches to en- code the spatial information as instead of their absolute position. To do so, we use 1- dimensional Relative Attention, in which we deﬁne the relative distance all possible pairs of patches. Thus, for every given pair (one as query, and the other as key/value in the at- tention mechanism), we have an offset  $p_{q}\mathrm{~-~}p_{k}$  , where each offset is associated with an embedding. Then, we simply run extra attention, where we use the original query (the content of query), but use relative positional embeddings as keys. We then use the log- its from the relative attention as a bias term and add it to the logits of the main attention (content-based attention) before applying the softmax.  

In addition to different ways of encoding spatial information, we also tried different ways of in- corporating this information in our model. For the 1-dimensional and 2-dimensional positional embeddings, we tried three different cases: (1) add positional embeddings to the inputs right after  

![](images/0b2abb9073bae1ca1beace909634f640fe091ec5b9a0d5fdf1b72bd91addbf5c.jpg)  
Figure 10: Position embeddings of models trained with different hyperparameters.  

the stem of them model and before feeding the inputs to the Transformer encoder (default across all other experiments in this paper); (2) learn and add positional embeddings to the inputs at the beginning of each layer; (3) add a learned positional embeddings to the inputs at the beginning of each layer (shared between layers).  

Table 8 summarizes the results from this ablation study on a ViT-B/16 model. As we can see, while there is a large gap between the performances of the model with no positional embedding and mod- els with positional embedding, there is little to no difference between different ways of encoding positional information. We speculate that since our Transformer encoder operates on patch-level inputs, as opposed to pixel-level, the differences in how to encode spatial information is less impor- tant. More precisely, in patch-level inputs, the spatial dimensions are much smaller than the original pixel-level inputs, e.g.,  $14\times14$   as opposed to  $224\times224$  , and learning to represent the spatial re- lations in this resolution is equally easy for these different positional encoding strategies. Even so, the speciﬁc pattern of position embedding similarity learned by the network depends on the training hyperparameters (Figure 10).  

![](images/98db1d307a741edbcb57a73870621f07421daaf019de804fcf5c1bbaae08d3b6.jpg)  
Figure 11: Size of attended area by head and network depth. Attention distance was computed for 128 example images by averaging the distance between the query pixel and all other pixels, weighted by the attention weight. Each dot shows the mean attention distance across images for one of 16 heads at one layer. Image width is 224 pixels.  

# D.5 E MPIRICAL  C OMPUTATIONAL  C OSTS  

We are also interested in real-world speed of the architectures on our hardware, which is not always well predicted by theoretical FLOPs due to details like lane widths and cache sizes. For this purpose, we perform timing of inference speed for the main models of interest, on a TPUv3 accelerator; the difference between inference and backprop speed is a constant model-independent factor.  

Figure 12 (left) shows how many images one core can handle per second, across various input sizes. Every single point refers to the peak performance measured across a wide range of batch-sizes. As can be seen, the theoretical bi-quadratic scaling of ViT with image size only barely starts happening for the largest models at the largest resolutions.  

Another quantity of interest is the largest batch-size each model can ﬁt onto a core, larger being better for scaling to large datasets. Figure 12 (right) shows this quantity for the same set of models. This shows that large ViT models have a clear advantage in terms of memory-efﬁciency over ResNet models.  

![](images/13b2295faf7bc82e1705b63fb3eafd726a9bc589e04033247a8439c7e6717bbc.jpg)  
Figure 12:  Left:  Real wall-clock timings of various architectures across input sizes. ViT models have speed comparable to similar ResNets.  Right : Largest per-core batch-size ﬁtting on device with various architectures across input sizes. ViT models are clearly more memory-efﬁcient.  

# D.6 A XIAL  A TTENTION  

Axial Attention (Huang et al., 2020; Ho et al., 2019) is a simple, yet effective technique to run self- attention on large inputs that are organized as multidimensional tensors. The general idea of axial attention is to perform multiple attention operations, each along a single axis of the input tensor, instead of applying 1-dimensional attention to the ﬂattened version of the input. In axial attention, each attention mixes information along a particular axis, while keeping information along the other axes independent. Along this line, Wang et al. (2020b) proposed the AxialResNet model in which all the convolutions with kernel size    $3\times3$   in a ResNet50 are replaced by axial self-attention, i.e. a row and column attention, augmented by relative positional encoding. We have implemented AxialResNet as a baseline model. .  

Moreover, we have modiﬁed ViT to process inputs in the 2-dimensional shape, instead of a 1- dimensional sequence of patches, and incorporate Axial Transformer blocks, in which instead of a self-attention followed by an MLP, we have a a row-self-attention plus an MLP followed by a column-self-attention plus an MLP.  

Figure 13, present the performance of Axial ResNet, Axial-ViT-B/32 and Axial-ViT-B/16 on Ima- geNet 5shot linear, when pretrained on JFT dataset, verses the pretraining compute, both in terms of number of FLOPs and inference time (example per seconds). As we can see, both Axial-ViT-B/32 and Axial-ViT-B/16 do better than their ViT-B counterpart in terms of performance, but it comes at  

![](images/6305a7e89505830935e3e06d3a76ce801beddb46fea725978b35068249278c97.jpg)  
Figure 13: Performance of Axial-Attention based models, in terms of top-1 accuracy on ImageNet 5-shot linear, versus their speed in terms of number of FLOPs ( left ) and inference time ( left ).  

the cost of more compute. This is because in Axial-ViT models, each Transformer block with global self-attention is replaced by two Axial Transformer blocks, one with row and one with column self- attention and although the sequence length that self-attention operates on is smaller in axial case, there is a extra MLP per Axial-ViT block. For the AxialResNet, although it looks reasonable in terms of accuracy/compute trade-off (Figure 13, left), the naive implementation is extremely slow on TPUs (Figure 13, right).  

# D.7 A TTENTION  D ISTANCE  

To understand how ViT uses self-attention to integrate information across the image, we analyzed the average distance spanned by attention weights at different layers (Figure 11). This “attention distance” is analogous to receptive ﬁeld size in CNNs. Average attention distance is highly variable across heads in lower layers, with some heads attending to much of the image, while others attend to small regions at or near the query location. As depth increases, attention distance increases for all heads. In the second half of the network, most heads attend widely across tokens.  

# D.8 A TTENTION  M APS  

To compute maps of the attention from the output token to the input space (Figures 6 and 14), we used Attention Rollout (Abnar & Zuidema, 2020). Brieﬂy, we averaged attention weights of ViT- L/16 across all heads and then recursively multiplied the weight matrices of all layers. This accounts for the mixing of attention across tokens through all layers.  

# D.9 O BJECT N ET  R ESULTS  

We also evaluate our ﬂagship ViT-H/14 model on the ObjectNet benchmark following the evaluation setup in Kolesnikov et al. (2020), resulting in  $82.1\%$   top-5 accuracy and   $61.7\%$   top-1 accuracy.  

# D.10 VTAB B REAKDOWN  

Table 9 shows the scores attained on each of the VTAB-1k tasks.  

![](images/99a101db889099b508f3c6b9f11897309f500672405fb2dbe59fd70ab18d5903.jpg)  
Figure 14: Further example attention maps as in Figure 6 (random selection).  

![Table 9: Breakdown of VTAB-1k performance across tasks. ](images/fcf8435bbdb9c2cd9d8238e3f813be71ebe960d5b84337a297cfa9618c2be2f0.jpg)  

![](images/df1ba96b518e0eb10a26b5c932a081fe271118bf74c42fdfc0611e7921a2a82e.jpg)  