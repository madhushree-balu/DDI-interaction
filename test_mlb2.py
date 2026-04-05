import sys
from sklearn.preprocessing import MultiLabelBinarizer

ALL_ORGANS = [
    'CARDIOVASCULAR', 'HEPATIC', 'RENAL', 'HEMATOLOGIC',
    'GASTROINTESTINAL', 'CENTRAL_NERVOUS_SYSTEM', 'RESPIRATORY',
    'ENDOCRINE', 'MUSCULOSKELETAL', 'IMMUNE_SYSTEM',
]

mlb = MultiLabelBinarizer(classes=ALL_ORGANS)
mlb.fit([['HEPATIC'], ['RENAL']])

with open('test_out.txt', 'w') as f:
    f.write(','.join(mlb.classes_) + '\n')
