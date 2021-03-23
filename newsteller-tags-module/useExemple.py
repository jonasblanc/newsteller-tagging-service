import category_tagging_module as tm
import pandas as pd

df = pd.read_parquet("data/nt_data_fr.gzip")
categories = ['sciences','economie','sport','culture', 'sante']
tm.train_save_model(df, categories, 'french')

title = "Restauration rapide – Le Covid fait plonger de 68 pourcent le bénéfice de MacDonald’s"
text = "Le géant de la restauration rapide McDonald’s a vu son bénéfice net chuter de 68 pourcent au deuxième trimestre et ne donne pas de prévision pour l’année, son activité ayant été drastiquement réduite par la pandémie partout dans le monde. McDonald’s a réalisé un bénéfice net de 483,8 millions de dollars (443,9 millions de francs) entre avril et juin."
category = tm.get_article_category(title, text , 'french')
print(category)

vector =  tm.get_article_vector(title, text , 'french')
print(vector)
