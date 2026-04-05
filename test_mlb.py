from sklearn.preprocessing import MultiLabelBinarizer
print(list(MultiLabelBinarizer(classes=['Z','A','C']).fit([['Z']]).classes_))
