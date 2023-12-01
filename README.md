# NLP-PaLM

This repository trains a Pathways Language Model (PaLM) based on the paper [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311). Like any Language Model, the model is trained in a self-supervised manner by predicting the next word, allowing it to learn a generative model to generate text.

To train the model using the [movie dialog dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), first run the script
```
python process_movie_dialogs_subword.py
```
to obtain the sub-word vocabulary. Then run
```
python train_movie_dialog_sw_tf_ver2_palm.py
```
or 
```
python train_movie_dialog_sw_torch_palm.py
```
to train the model in Tensorflow-2 or PyTorch respectively. After the model is trained, the script
```
python infer_movie_dialog_sw_tf_ver2_palm.py
```
or
```
python infer_movie_dialog_sw_torch_palm.py
```
can be run to perform inference of the trained model. 

Examples of the inferred model for Tensorflow-2 are:
```
Enter input phrase: what time is it?

Input Phrase:
what time is it?
Generated Response:
SOS two thirty . EOS
--------------------------------------------------
Enter input phrase: when are we leaving?

Input Phrase:
when are we leaving?
Generated Response:
SOS tomorrow . EOS
--------------------------------------------------
Enter input phrase: where are you going?

Input Phrase:
where are you going?
Generated Response:
SOS i ' m going to the bathroom . EOS
--------------------------------------------------
```

Examples of the inferred model for PyTorch are:
```
Enter input: what time is it?

Input Phrase:
what time is it?
Generated Reply:
SOS eight o ' clock . EOS
--------------------------------------------------
Enter input: when are we leaving?

Input Phrase:
when are we leaving?
Generated Reply:
SOS tomorrow . EOS 
--------------------------------------------------
Enter input: where are you going?

Input Phrase:
where are you going?
Generated Reply:
SOS i ' m not going to see my father . EOS 
--------------------------------------------------
```

