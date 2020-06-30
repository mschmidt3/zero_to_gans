# Move a Notebook to Colab

As I used up my kaggle GPU contigent  for this week, I tryed to run a notebook on kaggle.

The notebook used is: https://jovian.ml/mschmidt3/text-summarization-in-pytorch

Select [Run on colab] in the Run menu.

But starting the notebook on colab results in this error:
```
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-25-4b956591a88e> in <module>()
----> 1 data = pd.read_csv(data_path,encoding='utf-8')
      2 data.head()
...
```
