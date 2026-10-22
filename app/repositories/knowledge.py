"""In-memory disease metadata, shared by inference and history."""

DISEASE_METADATA = {
    "healthy": {"idx": 0, "name": "healthy", "disease": "Lá Khỏe Mạnh"},
    "miner": {"idx": 1, "name": "miner", "disease": "Sâu Đục Lá (Sâu Vẽ Bùa)"},
    "rust": {"idx": 2, "name": "rust", "disease": "Bệnh Gỉ Sắt Cà Phê"},
    "phoma": {"idx": 3, "name": "phoma", "disease": "Bệnh Đốm Lá Phoma (Lở Cổ Rễ / Khô Cành)"},
    "corticium": {"idx": 4, "name": "corticium", "disease": "Bệnh Nấm Hồng (Corticium Salmonicolor)"},
    "mealy": {"idx": 5, "name": "mealy", "disease": "Bệnh Rệp Sáp Hại Cà Phê"},
    "nematode": {"idx": 6, "name": "nematode", "disease": "Bệnh Tuyến Trùng Rễ / Vàng Lá Thối Rễ"},
    "anthracnose": {"idx": 7, "name": "anthracnose", "disease": "Bệnh Thán Thư Cà Phê (Khô Cành, Thối Quả)"},
}


def get_disease_knowledge() -> dict:
    return DISEASE_METADATA
