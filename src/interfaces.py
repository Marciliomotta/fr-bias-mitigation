from abc import ABC, abstractmethod

class IFaceModel(ABC):
    @abstractmethod
    def get_descriptors(self, image_path: str): pass
    @abstractmethod
    def compare(self, descriptor1, descriptor2) -> bool: pass
    @abstractmethod
    def get_name(self) -> str: pass