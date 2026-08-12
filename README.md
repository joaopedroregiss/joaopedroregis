# João Pedro Regis



import {
  AIProvider,
  AnalyzeRequest,
  ChatTurnRequest,
  ChatTurnResponse,
} from "./types";

import { AnalysisResult } from "@/lib/esg/types";
import { GoogleGenAI } from "@google/genai";

/* =========================================================
   CONFIGURAÇÃO
========================================================= */

const MODEL = "gemini-3.5-flash-lite";

const MAX_INTERVIEW_QUESTIONS = 8;

const READY_MESSAGE =
  "Entendi. Já tenho informações suficientes para analisar o potencial ESG dessa ideia.";

const interactionStore = new Map<string, string>();

/* =========================================================
   CLIENTE
========================================================= */

function getClient() {
  const apiKey = process.env.GEMINI_API_KEY?.trim();

  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY não configurada nas variáveis de ambiente."
    );
  }

  return new GoogleGenAI({
    apiKey,
  });
}

/* =========================================================
   ERRO DE LIMITE
========================================================= */

export class GeminiRateLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GeminiRateLimitError";
  }
}

function getErrorInfo(err: unknown) {
  const error = err as {
    status?: number;
    code?: number;
    message?: string;
    error?: {
      status?: number;
      code?: number;
      message?: string;
    };
  };

  const status =
    error?.status ??
    error?.code ??
    error?.error?.status ??
    error?.error?.code;

  const message =
    error?.message ??
    error?.error?.message ??
    String(err);

  return {
    status,
    message,
  };
}

function isRateLimitError(err: unknown): boolean {
  const { status, message } = getErrorInfo(err);

  return (
    status === 429 ||
    /429|RESOURCE_EXHAUSTED|quota|rate.?limit/i.test(message)
  );
}

/* =========================================================
   ID DA SESSÃO
========================================================= */

function getSessionKey(ideaText: string): string {
  return String(ideaText ?? "")
    .trim()
    .toLowerCase();
}

/* =========================================================
   INTERACTIONS API
========================================================= */

async function createInteraction({
  input,
  systemInstruction,
  previousInteractionId,
  json = false,
}: {
  input: string;
  systemInstruction: string;
  previousInteractionId?: string;
  json?: boolean;
}) {
  const ai = getClient();

  console.log(
    `[geminiProvider] Interactions API | model=${MODEL} | previousInteractionId=${previousInteractionId ? "SIM" : "NÃO"}`
  );

  try {
    const response = await ai.interactions.create({
      model: MODEL,

      ...(previousInteractionId
        ? {
            previous_interaction_id: previousInteractionId,
          }
        : {}),

      input,

      system_instruction: systemInstruction,

      response_format: json
        ? {
            type: "text",
            mime_type: "application/json",
          }
        : {
            type: "text",
          },
    });

    console.log(
      `[geminiProvider] Interação criada | id=${response.id}`
    );

    return response;
  } catch (err) {
    const { status, message } = getErrorInfo(err);

    console.error("[geminiProvider] ERRO NA INTERACTIONS API", {
      model: MODEL,
      status,
      message,
    });

    if (isRateLimitError(err)) {
      throw new GeminiRateLimitError(
        "Limite de uso do Gemini atingido. Verifique a cota do projeto no Google AI Studio."
      );
    }

    throw err;
  }
}

/* =========================================================
   EXTRAÇÃO DO TEXTO
========================================================= */

function extractText(response: any): string {
  if (typeof response?.text === "string") {
    return response.text.trim();
  }

  if (typeof response?.output_text === "string") {
    return response.output_text.trim();
  }

  const outputs = response?.outputs;

  if (Array.isArray(outputs)) {
    const textParts: string[] = [];

    for (const output of outputs) {
      if (typeof output?.text === "string") {
        textParts.push(output.text);
      }

      if (Array.isArray(output?.content)) {
        for (const content of output.content) {
          if (typeof content?.text === "string") {
            textParts.push(content.text);
          }
        }
      }
    }

    return textParts.join("\n").trim();
  }

  return "";
}

/* =========================================================
   ENTREVISTADOR
========================================================= */

const INTERVIEW_SYSTEM = `
Você é o entrevistador do Copiloto AEVO.

Sua função é compreender uma ideia de melhoria apresentada
por um colaborador.

A conversa é adaptativa.

Faça perguntas curtas, naturais e progressivas.

REGRAS OBRIGATÓRIAS:

- Faça SOMENTE UMA pergunta por vez.
- Responda SOMENTE com a pergunta.
- Não escreva "Olá".
- Não escreva "Entendi".
- Não escreva "Certo".
- Não explique sua lógica.
- Não faça listas.
- Não faça duas perguntas na mesma resposta.
- Não repita uma pergunta já respondida.
- Use principalmente a resposta mais recente para definir
  a próxima pergunta.
- Não invente informações.
- Não presuma que algo é ESG sem evidência.
- Não tente classificar ESG durante a entrevista.

Procure compreender:

- qual é o problema ou oportunidade;
- onde acontece;
- como funciona atualmente;
- quem participa ou é afetado;
- qual é a causa;
- o que o colaborador pretende mudar;
- qual resultado espera;
- qual impacto pode existir.

Se já houver informação suficiente para compreender
razoavelmente a ideia, responda EXATAMENTE:

"Entendi. Já tenho informações suficientes para analisar o potencial ESG dessa ideia."

Responda sempre em português do Brasil.
`;

/* =========================================================
   PROVIDER
========================================================= */

export const geminiProvider: AIProvider = {
  /* =======================================================
     PRÓXIMA PERGUNTA
  ======================================================= */

  async nextTurn({
    ideaText,
    answers,
  }: ChatTurnRequest): Promise<ChatTurnResponse> {
    const idea = String(ideaText ?? "").trim();

    if (!idea) {
      throw new Error("A ideia do colaborador não foi informada.");
    }

    if (answers.length >= MAX_INTERVIEW_QUESTIONS) {
      return {
        type: "ready",
        text: READY_MESSAGE,
      };
    }

    const sessionKey = getSessionKey(idea);

    const previousInteractionId =
      interactionStore.get(sessionKey);

    const questionNumber = answers.length + 1;

    let input: string;

    if (!previousInteractionId) {
      input = `
IDEIA ORIGINAL DO COLABORADOR:

${idea}

Esta é a primeira etapa da entrevista.

Faça a primeira pergunta necessária para compreender
melhor essa ideia.

Esta é a pergunta ${questionNumber} de no máximo ${MAX_INTERVIEW_QUESTIONS}.
`;
    } else {
      const latestAnswer =
        answers[answers.length - 1] ?? "";

      input = `
O COLABORADOR RESPONDEU:

${latestAnswer}

Faça agora a próxima pergunta necessária para aprofundar
essa resposta.

Esta é a pergunta ${questionNumber} de no máximo ${MAX_INTERVIEW_QUESTIONS}.

Não repita informações já obtidas.
`;
    }

    const response = await createInteraction({
      input,
      systemInstruction: INTERVIEW_SYSTEM,
      previousInteractionId,
    });

    if (response?.id) {
      interactionStore.set(
        sessionKey,
        response.id
      );
    }

    const text = extractText(response);

    if (!text) {
      throw new Error(
        "O Gemini retornou uma resposta vazia durante a entrevista."
      );
    }

    const cleanText = text
      .replace(/^["']|["']$/g, "")
      .trim();

    if (cleanText === READY_MESSAGE) {
      return {
        type: "ready",
        text: READY_MESSAGE,
      };
    }

    return {
      type: "question",
      text: cleanText,
    };
  },

  /* =======================================================
     ANÁLISE ESG
  ======================================================= */

  async analyze({
    ideaText,
    answers,
  }: AnalyzeRequest): Promise<AnalysisResult> {
    const idea = String(ideaText ?? "").trim();

    if (!idea) {
      throw new Error("A ideia do colaborador não foi informada.");
    }

    const history =
      answers.length > 0
        ? answers
            .map(
              (answer, index) =>
                `Resposta ${index + 1}: ${String(answer)
                  .replace(/\s+/g, " ")
                  .trim()}`
            )
            .join("\n")
        : "Nenhuma resposta fornecida.";

    const ANALYZE_SYSTEM = `
Você é o analista ESG do Copiloto AEVO.

Analise a ideia utilizando SOMENTE as informações
fornecidas pelo colaborador.

A conversa é a única fonte factual.

Você pode utilizar seu conhecimento de ESG para
organizar e classificar as informações.

NÃO INVENTE:

- impactos;
- benefícios;
- áreas;
- custos;
- números;
- processos;
- regulamentações;
- resultados;
- causas;
- consequências;
- indicadores.

DIMENSÕES ESG:

ENVIRONMENTAL:
água, energia, resíduos, materiais, descarte, emissões,
poluição, recursos naturais, reutilização, clima e
biodiversidade.

SOCIAL:
trabalhadores, saúde, segurança, condições de trabalho,
treinamento, capacitação, desenvolvimento, diversidade,
inclusão, acessibilidade, qualidade de vida, comunidade,
clientes e usuários.

GOVERNANCE:
processos, controles, procedimentos, políticas,
responsabilidades, registros, rastreabilidade,
transparência, auditoria, compliance, conformidade,
gestão de riscos, tomada de decisão e informações.

REGRA DE EVIDÊNCIA:

Só classifique uma dimensão quando houver evidência
concreta na conversa.

Se não houver evidência suficiente:

"NOT_IDENTIFIED"

BENEFÍCIOS:

Inclua somente benefícios mencionados pelo colaborador
ou consequências diretas e evidentes da mudança descrita.

ÁREAS:

Inclua somente áreas explicitamente mencionadas.

NÚMEROS:

Nunca invente valores, percentuais, quantidades,
economias, prazos ou metas.

PRÓXIMOS PASSOS:

Devem resolver lacunas reais da ideia.

Não adicione automaticamente auditoria, compliance,
indicadores, treinamento ou responsáveis.

FIDELIDADE À CONVERSA É MAIS IMPORTANTE QUE COMPLETUDE.

Retorne SOMENTE JSON válido.

Todo o conteúdo textual deve estar em português do Brasil.
`;

    const prompt = `
IDEIA ORIGINAL:

${idea}

HISTÓRICO DA ENTREVISTA:

${history}

Analise a ideia com base exclusivamente nas informações
acima.

Retorne exatamente esta estrutura:

{
  "status": "completed",
  "potential_esg": "HIGH",
  "dimensions": {
    "environmental": {
      "level": "NOT_IDENTIFIED",
      "justification": ""
    },
    "social": {
      "level": "NOT_IDENTIFIED",
      "justification": ""
    },
    "governance": {
      "level": "NOT_IDENTIFIED",
      "justification": ""
    }
  },
  "main_dimension": "environmental",
  "theme": "",
  "summary": "",
  "benefits": [],
  "areas": [],
  "next_steps": [],
  "mini_project": {
    "title": "",
    "description": ""
  }
}

Valores permitidos:

potential_esg:
"HIGH" | "MEDIUM" | "LOW"

dimensions.level:
"HIGH" | "MEDIUM" | "LOW" | "NOT_IDENTIFIED"

main_dimension:
"environmental" | "social" | "governance"

Não invente informações para preencher campos.
`;

    const response = await createInteraction({
      input: prompt,
      systemInstruction: ANALYZE_SYSTEM,
      json: true,
    });

    const raw = extractText(response);

    if (!raw) {
      throw new Error(
        "O Gemini retornou uma análise vazia."
      );
    }

    const cleaned = raw
      .replace(/^```json\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();

    try {
      return JSON.parse(cleaned) as AnalysisResult;
    } catch (error) {
      console.error(
        "[geminiProvider] JSON inválido retornado pelo Gemini:",
        {
          raw,
          error,
        }
      );

      throw new Error(
        "O Gemini retornou uma análise em formato inválido."
      );
    }
  },
};
