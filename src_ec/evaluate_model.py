def evaluate_model(label, predicted_label):
	from sklearn import metrics
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import cohen_kappa_score
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix
	conf_matrix=confusion_matrix(label,predicted_label)
	report=classification_report(label,predicted_label)
	acc_score=accuracy_score(label,predicted_label)
	ck_score=cohen_kappa_score(label,predicted_label)
	precision_micro=metrics.precision_score(label, predicted_label, average='micro')
	precision_macro=metrics.precision_score(label,predicted_label,average='macro')
	recall_micro=metrics.recall_score(label,predicted_label,average='micro')
	recall_macro=metrics.recall_score(label,predicted_label,average='macro')
	f1_micro=metrics.f1_score(label,predicted_label,average='micro')
	f1_macro=metrics.f1_score(label,predicted_label,average='macro')
	print("\n\n")
	print("The confusion matrix is as follows: \n")
	print(conf_matrix)
	print("\n\n")
	print("The classification result for each class is as follows: \n")
	print(report)
	print("\n\n")
	print("Here is the evaluation of the model performance: \n")
	print("The accuracy score is %f.\n"%acc_score)
	print("The Cohen's Kappa socre is %f.\n"%ck_score)
	print("The micro precistion is %f, the macro precision is %f.\n"%(precision_micro,precision_macro))
	print("The micro recall is %f, the macro recall is %f.\n"%(recall_micro,recall_macro))
	print("The micro F1 score is %f, the macro F1 score is %f.\n"%(f1_micro,f1_macro))




if __name__ == '__main__':
	pass