import os
import cv2
import numpy as np
from utils.config import FACES_DIR
from utils.logger import log


class FaceStore:

    def __init__(self, db_manager):
        self.db = db_manager
        self._cache_valid = False
        self._cached_encodings = []
        self._cached_names = []
        self._cached_ids = []

    def register_face(self, name, encodings, face_images, notes=""):
        try:
            person_id = self.db.add_person(name, notes)
            person_dir = os.path.join(FACES_DIR, str(person_id))
            os.makedirs(person_dir, exist_ok=True)

            for i, (encoding, face_img) in enumerate(zip(encodings, face_images)):
                img_path = os.path.join(person_dir, f"face_{i}.jpg")
                cv2.imwrite(img_path, face_img)

                encoding_bytes = encoding.tobytes()
                self.db.add_encoding(person_id, encoding_bytes, img_path)

            self._cache_valid = False
            log.info(f"Registered '{name}' with {len(encodings)} face encodings")
            return person_id

        except Exception as e:
            log.error(f"Failed to register face for '{name}': {e}")
            return None

    def load_all_faces(self):
        if self._cache_valid:
            return self._cached_encodings, self._cached_names, self._cached_ids

        encodings = []
        names = []
        ids = []

        rows = self.db.get_all_encodings()
        for row in rows:
            try:
                encoding = np.frombuffer(row["encoding"], dtype=np.float64)
                if encoding.shape[0] == 128:
                    encodings.append(encoding)
                    names.append(row["person_name"])
                    ids.append(row["person_id"])
            except Exception as e:
                log.warning(f"Skipping corrupt encoding ID {row['id']}: {e}")

        self._cached_encodings = encodings
        self._cached_names = names
        self._cached_ids = ids
        self._cache_valid = True

        log.info(f"Loaded {len(encodings)} face encodings for {len(set(names))} persons")
        return encodings, names, ids

    def invalidate_cache(self):
        self._cache_valid = False

    def delete_person(self, person_id):
        person_dir = os.path.join(FACES_DIR, str(person_id))
        if os.path.exists(person_dir):
            import shutil
            shutil.rmtree(person_dir)

        self.db.delete_person(person_id)
        self._cache_valid = False
        log.info(f"Deleted person ID {person_id} and their face data")

    def get_person_images(self, person_id):
        person_dir = os.path.join(FACES_DIR, str(person_id))
        if not os.path.exists(person_dir):
            return []

        images = []
        for fname in sorted(os.listdir(person_dir)):
            if fname.endswith((".jpg", ".png")):
                img_path = os.path.join(person_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((img_path, img))
        return images

    def get_person_thumbnail(self, person_id):
        images = self.get_person_images(person_id)
        if images:
            return images[0][1]
        return None
