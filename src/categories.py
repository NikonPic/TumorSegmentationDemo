def make_categories_advanced(simple=True, yolo=False):
    """fill the categories manually"""
    if simple:
        cat_list = [
            {
                "id": 0,
                "name": "Benign Tumor",
            },
            {
                "id": 1,
                "name": "Malignant Tumor",
            }
        ]
        if yolo:
            cat_mapping = {
                "benign": 0,
                "malign": 1,
            }
        else:
            cat_mapping = [0, 1]

        return cat_list, cat_mapping

    cat_list = [
        # malignant first
        {
            "supercategory": "Malignant",
            "id": 1,
            "name": "Chondrosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 2,
            "name": "Osteosarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 3,
            "name": "Ewing sarcoma",
        },
        {
            "supercategory": "Malignant",
            "id": 4,
            "name": "Plasma cell myeloma",
        },
        {
            "supercategory": "Malignant",
            "id": 5,
            "name": "NHL B Cell",
        },
        # now benign
        {
            "supercategory": "Benign",
            "id": 6,
            "name": "Osteochondroma",
        },
        {
            "supercategory": "Benign",
            "id": 7,
            "name": "Enchondroma",
        },
        {
            "supercategory": "Benign",
            "id": 8,
            "name": "Chondroblastoma",
        },
        {
            "supercategory": "Benign",
            "id": 9,
            "name": "Osteoid osteoma",
        },
        {
            "supercategory": "Benign",
            "id": 10,
            "name": "Non-ossifying fibroma",
        },
        {
            "supercategory": "Benign",
            "id": 11,
            "name": "Giant cell tumour of bone",
        },
        {
            "supercategory": "Benign",
            "id": 12,
            "name": "Chordoma",
        },
        {
            "supercategory": "Benign",
            "id": 13,
            "name": "Haemangioma",
        },
        {
            "supercategory": "Benign",
            "id": 14,
            "name": "Aneurysmal bone cyst",
        },
        {
            "supercategory": "Benign",
            "id": 15,
            "name": "Simple bone cyst",
        },
        {
            "supercategory": "Benign",
            "id": 16,
            "name": "Fibrous dysplasia",
        },
    ]
    # The names from datainfo are used here!
    cat_mapping = {
        # malign
        "Chondrosarkom": 1,
        "Osteosarkom": 2,
        "Ewing-Sarkom": 3,
        "Plasmozytom / Multiples Myelom": 4,
        "NHL vom B-Zell-Typ": 5,
        # benign
        "Osteochondrom": 6,
        "Enchondrom": 7,
        "Chondroblastom": 8,
        "Osteoidosteom": 9,
        "NOF": 10,
        "Riesenzelltumor": 11,
        "Chordom": 12,
        "Hämangiom": 13,
        "Knochenzyste, aneurysmatische": 14,
        "Knochenzyste, solitär": 15,
        "Dysplasie, fibröse": 16,
    }
    return cat_list, cat_mapping

reverse_cat_list = [
    "Knochenzyste, aneurysmatische",
    "Enchondrom",
    "Dysplasie, fibröse",
    "Knochenzyste, solitär",
    "Osteochondrom",
    "NHL vom B-Zell-Typ",
    "Plasmozytom / Multiples Myelom",
    "Ewing-Sarkom",
    "NOF",
    "Chordom",
    "Chondroblastom",
    "Osteosarkom",
    "Hämangiom",
    "Chondrosarkom",
    "Osteoidosteom",
    "Riesenzelltumor",
]