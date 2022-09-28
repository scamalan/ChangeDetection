# ChangeDetection
Abstract: Monitoring changes within the land surface and open water bodies is critical for natural
resource management, conservation, and environmental policy. While the use of satellite imagery
for these purposes is common, fine-scale change detection can be a technical challenge. Difficulties
arise from variable atmospheric conditions and the problem of assigning pixels to individual objects.
We examined the degree to which two machine learning approaches can better characterize change
detection in the context of a current conservation challenge, artisanal small-scale gold mining (ASGM).
We obtained Sentinel-2 imagery and consulted with domain experts to construct an open-source
labeled land-cover change dataset. The focus of this dataset is the Madre de Dios (MDD) region in
Peru, a hotspot of ASGM activity. We also generated datasets of active ASGM areas in other countries
(Venezuela, Indonesia, and Myanmar) for out-of-sample testing. With these labeled data, we utilized
a supervised (E-ReCNN) and semi-supervised (SVM-STV) approach to study binary and multi-class
change within mining ponds in the MDD region. Additionally, we tested how the inclusion of
multiple channels, histogram matching, and La*b* color metrics improved the performance of the
models and reduced the influence of atmospheric effects. Empirical results show that the supervised
E-ReCNN method on 6-Channel histogram-matched images generated the most accurate detection
of change not only in the focal region (Kappa: 0.92 (± 0.04), Jaccard: 0.88 (± 0.07), F1: 0.88 (± 0.05))
but also in the out-of-sample prediction regions (Kappa: 0.90 (± 0.03), Jaccard: 0.84 (± 0.04), and F1:
0.77 (± 0.04)). While semi-supervised methods did not perform as accurately on 6- or 10-channel
imagery, histogram matching and the inclusion of La*b* metrics generated accurate results with low
memory and resource costs. These results show that E-ReCNN is capable of accurately detecting
specific and object-oriented environmental changes related to ASGM. E-ReCNN is scalable to areas
outside the focal area and is a method of change detection that can be extended to other forms of
land-use modificatio
