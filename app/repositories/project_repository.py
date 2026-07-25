from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[Project]:
        result = await self.session.execute(select(Project).order_by(Project.created_at.desc()))
        return list(result.scalars().all())

    async def create(self, name: str, description: str | None) -> Project:
        project = Project(name=name, description=description)
        self.session.add(project)
        await self.session.flush()
        return project
