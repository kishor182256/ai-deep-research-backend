from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Project(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    suggestion_batches: Mapped[list["ResearchSuggestionBatch"]] = relationship(back_populates="project")
    jobs: Mapped[list["ResearchJob"]] = relationship(back_populates="project")
