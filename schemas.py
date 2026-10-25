from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RespuestaLLM(BaseModel):
    """Lo que genera el LLM. Lo parsea el PydanticOutputParser a partir del JSON de la respuesta.

    El modelo NO escribe las referencias: solo indica qué fragmentos numerados usó. Las referencias
    (documento, punto, página, vigencia) las arma el código con los metadatos reales de esos
    fragmentos, así no puede inventar citas.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta a la pregunta basada EXCLUSIVAMENTE en el CONTEXTO, en español. "
            "Si el contexto no alcanza para responder, decir 'No lo sé' y explicar brevemente qué falta."
        ),
    )
    encontrada: bool = Field(
        description="true si el CONTEXTO contiene la información para responder; false en caso contrario.",
    )
    fragmentos_usados: List[int] = Field(
        default_factory=list,
        description="Números de los fragmentos del CONTEXTO que respaldan la respuesta (ej. [1, 3]). Vacío si encontrada=false.",
    )

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaLLM":
        if not self.encontrada:
            # Si no hay respuesta, no puede haber fuentes que la respalden
            self.fragmentos_usados = []
        elif not self.fragmentos_usados:
            # Una respuesta sin fragmentos que la respalden no es verificable: se rechaza (y se reintenta)
            raise ValueError("encontrada=true requiere indicar al menos un fragmento en fragmentos_usados")
        self.fragmentos_usados = sorted(set(self.fragmentos_usados))
        return self


class Referencia(BaseModel):
    """Cita verificable a la normativa, construida desde los metadatos del chunk."""

    documento: str
    seccion: str
    punto: str
    paginas: str = Field(description="Página(s) del PDF original, ej. '7' o '7-8'")
    vigencia: str = Field(description="Fecha de vigencia de esa página según el pie del texto ordenado")
    url: str
    relevancia: float = Field(ge=0, le=1, description="Similitud coseno entre la pregunta y el fragmento")


class RespuestaRAG(BaseModel):
    """Objeto final que devuelve get_rag_response()."""

    pregunta: str
    respuesta: str
    encontrada: bool
    referencias: List[Referencia] = Field(default_factory=list)
    fragmentos_recuperados: int = Field(description="Fragmentos que superaron el umbral de relevancia")
    descartada_por_umbral: bool = Field(
        default=False,
        description="true si ningún fragmento superó el umbral y se respondió sin llamar al LLM",
    )
