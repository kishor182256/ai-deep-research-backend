from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.project_repository import ProjectRepository
from app.schemas.projects import ProjectCreate, ProjectRead

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectRead])
async def list_projects(session: AsyncSession = Depends(get_db_session)) -> list[ProjectRead]:
    projects = await ProjectRepository(session).list()
    return [
        ProjectRead(id=project.id, name=project.name, description=project.description)
        for project in projects
    ]


@router.post("", response_model=ProjectRead)
async def create_project(
    payload: ProjectCreate,
    session: AsyncSession = Depends(get_db_session),
) -> ProjectRead:
    project = await ProjectRepository(session).create(
        name=payload.name,
        description=payload.description,
    )
    return ProjectRead(id=project.id, name=project.name, description=project.description)
