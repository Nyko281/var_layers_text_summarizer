# Variable Layer Fine-Tuning for NLP Summarization Tasks

The GPT model used for this paper works with twelve layers. Regularly, such language models are trained or fine-tuned by taking a dataset with examples as input and applying it to part of the layers, while the remaining layers are not changed. This preserves the general language information of the pre-training. For example, only the last output layer is trained. <br>
In this work, on the other hand, two independent training sets will be used to train different layers. For this purpose, it was tested beforehand which and how many layers should be trained with which data set. Afterwards a finished model is trained.
