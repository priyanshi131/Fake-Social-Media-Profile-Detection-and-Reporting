from keras.models import Sequential,model_from_json
import pandas as pd

# load json and create model

# Write the file name of the model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")



# insert / provide input data here
prediction_df = pd.DataFrame([{"statuses_count" : 2,
    "followers_count":214,
    "friends_count":367,
    "favourites_count":6,
    "lang_num":1,
    "listed_count":25,
    "geo_enabled":0,
    "profile_use_background_image":0}])

print(prediction_df)


prediction = loaded_model.predict(prediction_df)
prediction = prediction[0]
print('Prediction\n',prediction)
# print('\nThresholded output\n',(prediction>0.5)*1)
if prediction > 0.5:
    print("fake profile")
else:
    print("real profile")



