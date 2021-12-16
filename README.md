# covid-ct-seg

## Contour-enhanced Attention CNN for CT-based COVID-19 Segmentation

**Contributors:** Hariharan, M., Karthik, R., Menaka, R.

### Highlights
- Proposed Contour enhancement to refine attention focus areas in CNN learning to find lesions on CT scan.
- Exploiting contour features was facilitated through a comprehensive cross-attention mechanism that fuses lower and higher order features for upsampling and decoding.

Accurate detection of COVID-19 is one of the challenging research topics in today’s healthcare sector to control the coronavirus pandemic. Automatic data-powered insights for COVID-19 localization from medical imaging modality like chest CT scan tremendously augments clinical care assistance. In this research, a Contour-aware Attention Decoder CNN has been proposed to precisely segment such COVID-19 infected tissues in a very effective way. It introduces a novel attention scheme to extract boundary, shape cues from CT contours and leverage those features in refining the infected areas. Specifically, for every decoded pixel, the attention module harvests contextual information of pixels in its spatial neighborhood from the contour feature maps. As a result of incorporating such rich structural details into decoding via dense attention, the CNN is able to capture even intricate infection morphology and minuscule islands. Besides, the decoder is augmented with a Cross Context Attention Fusion Upsampling to robustly reconstruct deep semantic features back to high-resolution segmentation map. This upsampler employs a novel pixel-precise attention model that draws relevant encoder features to aid in upsampling. The proposed CNN was evaluated on 3D scans from MosMedData and Jun Ma benchmarked datasets. It achieved state-of-the-art performance with a high dice similarity coefficient of 85.43% and a recall of 88.10%.
