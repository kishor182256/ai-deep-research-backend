from app.core.config import settings
from app.repositories.research_repository import ResearchRepository
from app.services.model_router import ModelRoute


class CostTrackerService:
    async def record_model_call(
        self,
        *,
        repository: ResearchRepository,
        job_id: str | None,
        task_type: str,
        route: ModelRoute,
        input_text: str = "",
        output_text: str = "",
        reason_suffix: str | None = None,
    ) -> None:
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        estimated_cost = self.estimate_model_cost(
            model=route.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        reason = route.reason if not reason_suffix else f"{route.reason} {reason_suffix}"

        await repository.create_model_call_log(
            job_id=job_id,
            provider=route.provider,
            model=route.model,
            task_type=task_type,
            reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
        )

    async def record_cost(
        self,
        *,
        repository: ResearchRepository,
        job_id: str | None,
        category: str,
        amount: float = 0.0,
        description: str | None = None,
    ) -> None:
        await repository.create_cost_record(
            job_id=job_id,
            category=category,
            amount=round(max(amount, 0), 6),
            description=description,
        )

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_model_cost(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        total_tokens = input_tokens + output_tokens
        if total_tokens == 0:
            return 0.0

        rate = (
            settings.estimated_fast_model_cost_per_1k_tokens
            if model == settings.default_fast_model
            else settings.estimated_reasoning_model_cost_per_1k_tokens
        )
        return round((total_tokens / 1000) * rate, 6)

    def estimate_search_cost(self, *, call_count: int) -> float:
        return round(call_count * settings.estimated_search_cost_per_call, 6)
