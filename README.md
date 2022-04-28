# NLP_pro
Sentiment analysis for reviews about Freedom_Convoy from reddit\
Multi-class classification task\
Note: pay much attention on the match of the one-hot label and the predictions\
predict == labels could lead to mathching errors when discuss accuracy\
--->\
preds = torch.round(logits) \ 
p = (preds - labels).cpu().numpy() \
num = np.count_nonzero(p, axis = 1)\
zero_num = num == 0\
accuracy = zero_num.mean()\
