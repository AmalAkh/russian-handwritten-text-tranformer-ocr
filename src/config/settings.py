class Settings:
    
    EPOCHS:int=10
    BATCH_SIZE:int=16
    RANDOM_STATE:int=42
    NUM_WORKERS:int = 8

    TRAIN_SPLIT:float=0.65
    TEST_SPLIT:float=0.15
    VAL_SPLIT:float=0.05

    LOSS_FUNCTION:str = "CrossEntropy"

    RESIZE_IMAGE = (224,224)

    PRETRAINED_VIT_ENCODER = "google/vit-huge-patch14-224-in21k"
    D_MODEL = 1280
    TRAIN_VIT_ENCODER = True



    @classmethod
    def to_dict(cls) -> dict:
        return {
            "EPOCHS": cls.EPOCHS,
            "BATCH_SIZE": cls.BATCH_SIZE,
            "RANDOM_STATE": cls.RANDOM_STATE,
            "NUM_WORKERS": cls.NUM_WORKERS,
            "TRAIN_SPLIT": cls.TRAIN_SPLIT,
            "TEST_SPLIT": cls.TEST_SPLIT,
            "VAL_SPLIT": cls.VAL_SPLIT,
            "LOSS_FUNCTION": cls.LOSS_FUNCTION,
            "RESIZE_IMAGE": cls.RESIZE_IMAGE
        }