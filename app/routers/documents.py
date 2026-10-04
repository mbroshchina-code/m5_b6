"""Административная загрузка карточек багов в базу знаний."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import structlog
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from filelock import FileLock, Timeout

from app.admin.auth import require_admin
from app.core.config import get_settings
from app.services.ingestion import IngestionService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(require_admin)],
)
log = structlog.get_logger(__name__)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def index_uploaded_document(path: Path, job_id: str) -> None:
    """Фоновая задача: запуск существующего ingestion pipeline."""
    try:
        service = IngestionService()
        service.state_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "document_ingestion_started",
            job_id=job_id,
            file_name=path.name,
        )

        # Несколько HTTP-загрузок индексируем последовательно.
        # Внутри service.run есть отдельная общая блокировка ingestion.
        with FileLock(
            str(service.state_dir / "upload_worker.lock"),
            timeout=600,
        ):
            result = service.run(path)

        log.info(
            "document_ingestion_completed",
            job_id=job_id,
            file_name=path.name,
            **result,
        )

    except Exception as exc:
        # Не выводим содержимое документа, ключи и текст ответа провайдера.
        log.error(
            "document_ingestion_failed",
            job_id=job_id,
            file_name=path.name,
            error_type=type(exc).__name__,
        )


@router.post(
    "/upload",
    status_code=202,
    summary="Загрузить карточку бага в базу знаний",
    responses={
        202: {"description": "Файл сохранён, индексация поставлена в фон"},
        400: {"description": "Неправильное имя или содержимое файла"},
        401: {"description": "Неверный административный токен"},
        409: {"description": "Карточка с этим номером уже существует"},
        413: {"description": "Файл превышает допустимый размер"},
        415: {"description": "Неподдерживаемый формат"},
        503: {"description": "Загрузка временно недоступна"},
    },
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    settings = get_settings()
    filename = file.filename or ""

    # Строгий формат имени одновременно сохраняет bug_id
    # и запрещает передавать пути вместо имени файла.
    match = re.fullmatch(r"bug_([1-9]\d*)\.(pdf|md)", filename)

    if not match:
        raise HTTPException(
            status_code=400,
            detail="Имя файла должно иметь вид bug_162.pdf или bug_162.md",
        )

    category = settings.documents_upload_category

    if (
        not re.fullmatch(r"[a-z0-9_-]+", category)
        or category not in settings.ingest_categories
    ):
        raise HTTPException(
            status_code=503,
            detail="Категория загрузки не настроена для индексации",
        )

    service = IngestionService()
    directory = service.data_root / category
    directory.mkdir(parents=True, exist_ok=True)

    if not directory.resolve().is_relative_to(service.data_root):
        raise HTTPException(
            status_code=503,
            detail="Некорректный каталог загрузки",
        )

    target = directory / filename
    temporary = None

    try:
        size = 0

        # Незавершённые загрузки имеют расширение .part:
        # ingestion не принимает их за готовые документы.
        with NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix="upload_",
            suffix=".part",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)

            while block := file.file.read(1024 * 1024):
                size += len(block)

                if size > settings.documents_upload_max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Максимальный размер файла — 10 МБ",
                    )

                stream.write(block)

        if size == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")

        if target.suffix == ".pdf":
            with temporary.open("rb") as stream:
                header = stream.read(1024)

            if b"%PDF-" not in header:
                raise HTTPException(
                    status_code=415,
                    detail="Содержимое файла не похоже на PDF",
                )
        else:
            try:
                text = temporary.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Markdown необходимо сохранить в UTF-8",
                ) from None

            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Markdown не содержит текста",
                )

        service.state_dir.mkdir(parents=True, exist_ok=True)

        with FileLock(
            str(service.state_dir / "upload_save.lock"),
            timeout=5,
        ):
            # Не создаём две карточки одного бага в разных форматах
            # внутри разрешённых категорий.
            for allowed_category in settings.ingest_categories:
                for extension in (".md", ".pdf"):
                    existing = (
                        service.data_root
                        / allowed_category
                        / f"{target.stem}{extension}"
                    )

                    if existing.is_symlink():
                        raise HTTPException(
                            status_code=409,
                            detail="Путь карточки является символической ссылкой",
                        )

                    if existing.exists() and existing != target:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Карточка этого бага уже существует "
                                "в другой категории или другом формате"
                            ),
                        )

            already_saved = target.exists()

            if already_saved:
                if file_hash(target) != file_hash(temporary):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Файл с этим именем уже существует и отличается. "
                            "Автоматическая перезапись запрещена."
                        ),
                    )
            else:
                os.replace(temporary, target)
                temporary = None

        job_id = uuid4().hex

        background_tasks.add_task(
            index_uploaded_document,
            target,
            job_id,
        )

        return {
            "status": "accepted",
            "job_id": job_id,
            "file_name": filename,
            "category": category,
            "collection": service.collection,
            "already_saved": already_saved,
        }

    except Timeout:
        raise HTTPException(
            status_code=503,
            detail="Другая загрузка сохраняет файл. Повторите позже.",
        ) from None

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

        file.file.close()