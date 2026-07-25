from sqlalchemy import ForeignKey, Integer, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class ModelCallLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_call_logs"

    job_id: Mapped[str | None] = mapped_column(ForeignKey("research_jobs.id"), nullable=True)
    provider: Mapped[str] = mapped_column(nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    task_type: Mapped[str] = mapped_column(nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost: Mapped[float] = mapped_column(Numeric(10, 6), default=0, nullable=False)


class CostRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "cost_records"

    job_id: Mapped[str | None] = mapped_column(ForeignKey("research_jobs.id"), nullable=True)
    category: Mapped[str] = mapped_column(nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(10, 6), default=0, nullable=False)
    currency: Mapped[str] = mapped_column(default="USD", nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
