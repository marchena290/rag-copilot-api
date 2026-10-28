from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from datetime import datetime, timezone

from database import Base


class HistorialRAG(Base):
    __tablename__ = "historial_rag"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, nullable=True)
    ticket_id = Column(Integer, nullable=True)
    modo = Column(String(20), default="libre")
    consulta = Column(Text, nullable=False)
    respuesta = Column(Text, nullable=False)
    fuentes = Column(JSON, default=list)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
