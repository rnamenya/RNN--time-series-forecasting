# RNN--time-series-forecasting

# Time-Series Forecasting with Recurrent Neural Network (RNN)

![image](https://github.com/user-attachments/assets/4b47576b-d1b0-4725-97ed-74d0d2f95c8f)



## 📌 Overview
This project demonstrates how to use a Recurrent Neural Network (RNN) to analyze and forecast time-series data. The model is designed to uncover temporal patterns and make accurate predictions that can inform strategic business decisions.

## 📊 Objective
To build a time-series prediction model using RNN (LSTM variant) in Python that can forecast future values based on historical data trends.

## 🧠 Key Techniques
- Recurrent Neural Networks (RNN)
- LSTM (Long Short-Term Memory) Networks
- Data preprocessing and normalization
- Time-series decomposition

I used python on jupiter note book

## 🎥 Project Presentation (YouTube)

For a complete walkthrough of this time-series forecasting project — including objectives, methods, model results, and business insights — watch the video below:
▶️ [Watch on YouTube](https://youtu.be/3g52xwz61E0)

Below is my presentation of the project 

![image](https://github.com/user-attachments/assets/8c9fe5ee-9d9a-40b0-9548-aee17ce048bc)

The Data Set
![image](https://github.com/user-attachments/assets/0fe8de18-0dec-4b6d-ac42-9a3986d6986c)

Data Source: Kaggle 
https://www.kaggle.com/datasets/shenba/time-series-datasets?select=monthly-beer-production-in-austr.csv
![image](https://github.com/user-attachments/assets/14baedcb-df4d-4aaf-8ee4-31ca3ba4d5e9)

The first step was to build a simple RNN with Sine Wave data
![image](https://github.com/user-attachments/assets/3f4da1a9-b5e9-4038-af66-74280b415d2a)
![image](https://github.com/user-attachments/assets/a8541b04-2051-4515-82ef-c0d7cfab000d)
![image](https://github.com/user-attachments/assets/04ee9dbf-2d50-48f4-940c-8388f85f5aed)
![image](https://github.com/user-attachments/assets/65104538-52a9-4677-857c-7bcb67d5640a)
![image](https://github.com/user-attachments/assets/52fd2f64-a68b-42be-b21b-aa638c46d825)
![image](https://github.com/user-attachments/assets/257d2bf8-65f1-4ee1-869e-9d7f9836dff3)
![image](https://github.com/user-attachments/assets/d2cbb02f-c4de-459a-b967-73283304b540)
![image](https://github.com/user-attachments/assets/70052d6a-df48-4cf4-adbb-c7d540c3c070)
![image](https://github.com/user-attachments/assets/6aca371b-0e38-4085-a8c1-8308e2c1fd4e)
![image](https://github.com/user-attachments/assets/76da3735-6763-42dc-a63f-2fe825bac625)
![image](https://github.com/user-attachments/assets/8fef6e57-3523-4590-83b8-8fd686f75292)
![image](https://github.com/user-attachments/assets/5c702123-1646-43f1-ae50-7882e348dee1)
![image](https://github.com/user-attachments/assets/16364f0c-c23c-4ed0-8325-7cbf227912e2)

The next step is explaining the vanishing gradient problem, solutions and the limitation of the Simple RNN neural network  -RNN: LSTM Neural Network 

![image](https://github.com/user-attachments/assets/9014ae4e-bcae-418e-a70b-70b45aaaea6d)

![image](https://github.com/user-attachments/assets/5ee7c906-369b-4778-9e9f-17628b901b5b)

![image](https://github.com/user-attachments/assets/8a0c60fe-4c10-495a-81e0-0777df17b3fd)

![image](https://github.com/user-attachments/assets/9feae623-aeef-494f-84db-234d216dd543)

![image](https://github.com/user-attachments/assets/5d7196b0-71da-4e53-92e3-4f58c243c811)

The step that follows involves preprocessing the original data to collect a time series data set with atleast 500 data points and then ensuring the below 
--) Building an LSTM neural network using the Keras sequential model 
--) Training the model using the training data, part of the collected data set. 
--) Testing the model using the testing data that has been allocated from the collected data set. 
--) Retraining the model using the whole collected data set and prepare it for forecasting. 
--) Performing the forecasting using the retrained model. 
--) Running all the steps of the project in a Jupyter Notebook document to get the results of each step. 

Below is an overview presentation of what i did 
![image](https://github.com/user-attachments/assets/3e3af9c7-a39c-4f5b-8fb3-ae6f0836d452)

![image](https://github.com/user-attachments/assets/c288ad9f-ec6c-4781-b152-615baebdde2d)

![image](https://github.com/user-attachments/assets/d5d43ede-19e6-4c30-afc5-c5c6df552270)
![image](https://github.com/user-attachments/assets/8ceb8544-ff70-426d-8cb6-653b70078343)

![image](https://github.com/user-attachments/assets/42a3d4f5-5c8b-432d-bc76-15e45351c55f)

![image](https://github.com/user-attachments/assets/6f464bb3-4429-47c6-8702-b146ed421374)

![image](https://github.com/user-attachments/assets/3536c2e4-a508-4e1f-a536-bf919d7c97f9)

![image](https://github.com/user-attachments/assets/b303ba34-c547-45fd-bc9c-c4fe4c0b13a2)

![image](https://github.com/user-attachments/assets/0c9dc9e5-ac01-4286-848f-9f0dd9086e15)

![image](https://github.com/user-attachments/assets/3074a499-6492-402b-9bc3-265be78e4ed2)

![image](https://github.com/user-attachments/assets/0d10ba9d-c0b0-4c5f-a977-9ec2bbd1297c)

![image](https://github.com/user-attachments/assets/c1c79d97-ed4d-4ac6-b016-3cf127de2090)

![image](https://github.com/user-attachments/assets/672c3c2c-8fc3-4d9c-b5c0-aa079ba563a1)

The final step i did was to compare network performance based on the design and the core parameters applied to the LSTM neural network in PART III and 
the network in PART IV. Below is an overview 

![image](https://github.com/user-attachments/assets/c9930b21-234f-4a31-8be3-1049a2375925)

![image](https://github.com/user-attachments/assets/6952b187-5ea6-48e6-ab74-0e7f961787f3)

![image](https://github.com/user-attachments/assets/6537693c-645a-4ee1-ae91-8c1fe18fe75e)

![image](https://github.com/user-attachments/assets/9398be20-5001-4ae9-8083-ba3fc47b6886)

![image](https://github.com/user-attachments/assets/5aa2402f-f188-4ee5-8024-bfe69420a518)

![image](https://github.com/user-attachments/assets/019fd5e9-37c5-4022-b444-6c90c56d6c3e)

![image](https://github.com/user-attachments/assets/051b0a03-14e5-4dab-a697-c3d58013a260)





















