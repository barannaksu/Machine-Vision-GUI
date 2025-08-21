import os
import torch
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine


def main():
    os.environ["ANOMALIB_SKIP_IMPORT_OPENVINO"] = "1"

    datamodule = MVTec(
        root="MVTec",
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=15
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    torch.save(model.model.state_dict(), "padim_bottle.pth")


if __name__ == "__main__":
    main()

