"""Face detection/alignment (MTCNN) and 512-d embeddings (FaceNet, VGGFace2)."""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps

_device = "cuda" if torch.cuda.is_available() else "cpu"
_detector = None
_embedder = None


def _load():
    global _detector, _embedder
    if _detector is None:
        _detector = MTCNN(image_size=160, margin=20, keep_all=False, device=_device)
        _embedder = InceptionResnetV1(pretrained="vggface2").eval().to(_device)


def embed_image(img: Image.Image, flip_augment=False):
    """Return a list of embeddings (1, or 2 with a mirrored copy), or [] if no face."""
    _load()
    img = ImageOps.exif_transpose(img).convert("RGB")
    imgs = [img, ImageOps.mirror(img)] if flip_augment else [img]
    out = []
    for im in imgs:
        face = _detector(im)
        if face is None:
            continue
        with torch.no_grad():
            out.append(_embedder(face.unsqueeze(0).to(_device)).cpu().numpy()[0])
    return out
