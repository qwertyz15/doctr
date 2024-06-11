import torch
from doctr.models import ocr_predictor, db_resnet50, parseq, crnn_vgg16_bn, vitstr_small

english_vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
bengali_vocab = 'ঁংঅইউএওকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুেোৌ্য়০১২৩৪৫৬৭৮৯- '
# Load custom detection and recognition model
det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
det_params = torch.load('/home/dev/Documents/doctr/db_resnet50_20240610-151622_word_sigmind_data.pt', map_location="cpu")
det_model.load_state_dict(det_params)
reco_model = parseq(pretrained=False, pretrained_backbone=False, vocab=bengali_vocab)

reco_params = torch.load("/home/dev/Documents/doctr/parseq_20240606-141938.pt", map_location="cpu")
reco_model.load_state_dict(reco_params)
predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False)


from doctr.io import DocumentFile
doc = DocumentFile.from_images("/home/dev/Documents/doctr/DETECTION_DATASETS/detection_dataset/val/images/4410443_png.rf.03693193ac49276356a8c29390ffff08.jpg")
result = predictor(doc)
print(result)
