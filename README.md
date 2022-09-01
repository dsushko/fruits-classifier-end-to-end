# fruits-classifier-end-to-end

This is my Personal Development project

This app can classify pictures accodringly to what fruit or vegetable they contain.
Currently they're apple, banana, beetroot, bell pepper, cabbage, carrot, cauliflower,
chili pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalapeno, kiwi,
lemon, lettuce, mango, onion, orange, pear, peas, pineapple, pomegranate,
potato, raddish, spinach, tomato, turnip, watermelon.

Model runs with

```
python run.py %build% %action% (-v) (-p)
```
```build``` - model to fit and predict, currently there are ```baseline``` (SVM classifer) and ```vgg16``` (VGG16 neural network model architecture Imagenet-based) options
```action``` - one of ```train```, ```predict```, ```explainability```
```-v``` - if ```action=predict```, model validation will be performed (validation score & confusion matrix)
```-p``` - if ```action=explainability```, then this parameter introduces path to image to be explained (should follow after this ```-p``` argument)

Data is stored in ./data/train and ./data/test folder
