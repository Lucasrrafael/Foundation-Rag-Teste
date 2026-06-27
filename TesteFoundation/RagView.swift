//
//  RagView.swift
//  TesteFoundation
//

import SwiftUI

struct RagView: View {
    @State private var engine = RagEngine()
    @FocusState private var isQuestionFocused: Bool

    @State private var question = "O que é abuso sexual contra crianças?"
    @State private var answer: RagAnswer?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Group {
                #if targetEnvironment(simulator)
                simulatorUnavailableView
                #else
                mainContent
                #endif
            }
            .navigationTitle("RAG")
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Concluído") {
                        dismissKeyboard()
                    }
                }
            }
            .task {
                if case .idle = engine.state {
                    await engine.indexDocuments()
                }
            }
        }
    }

    @ViewBuilder
    private var mainContent: some View {
        switch engine.state {
        case .idle, .indexing:
            VStack(spacing: 16) {
                ProgressView()
                Text("Indexando PDFs…")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

        case .loadingModel(let progress):
            VStack(spacing: 16) {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(maxWidth: 280)
                Text("Baixando modelo Qwen2.5-3B…")
                    .font(.headline)
                Text("\(Int(progress * 100))%")
                    .foregroundStyle(.secondary)
                Text("Na primeira execução, o download pode levar alguns minutos.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

        case .ready(let passageCount, let mode):
            questionForm(passageCount: passageCount, mode: mode)

        case .answering:
            questionForm(passageCount: nil, mode: nil, isAnswering: true)

        case .error(let message):
            ContentUnavailableView(
                "Erro no RAG",
                systemImage: "exclamationmark.triangle",
                description: Text(message)
            )
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Tentar novamente") {
                        Task { await engine.retrySetup() }
                    }
                }
            }
        }
    }

    private func questionForm(
        passageCount: Int?,
        mode: RagRetrievalMode?,
        isAnswering: Bool = false
    ) -> some View {
        Form {
            if let passageCount, let mode {
                Section("Índice") {
                    LabeledContent("Passagens indexadas", value: "\(passageCount)")
                    LabeledContent("Modo de busca", value: mode.rawValue)
                    LabeledContent("Modelo de geração", value: "Qwen2.5-3B (MLX)")
                }
            }

            Section("Pergunta sobre as cartilhas") {
                TextField("Digite sua pergunta…", text: $question, axis: .vertical)
                    .lineLimit(3...6)
                    .focused($isQuestionFocused)
                    .submitLabel(.send)
                    .onSubmit {
                        Task { await askQuestion() }
                    }

                Button("Perguntar") {
                    dismissKeyboard()
                    Task { await askQuestion() }
                }
                .disabled(question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isAnswering)

                if isAnswering {
                    ProgressView("Buscando contexto e gerando resposta…")
                }
            }

            if let errorMessage {
                Section("Erro") {
                    Text(errorMessage)
                        .foregroundStyle(.red)
                }
            }

            if let answer {
                Section("Resposta") {
                    Text(answer.response)
                        .textSelection(.enabled)
                }

                Section("Fontes usadas") {
                    ForEach(answer.sources) { source in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(source.sourceTitle)
                                .font(.headline)
                            Text(source.excerpt)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            Text(String(format: "Relevância: %.2f", source.score))
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .scrollDismissesKeyboard(.immediately)
    }

    private var simulatorUnavailableView: some View {
        ContentUnavailableView(
            "Simulador não suportado",
            systemImage: "iphone",
            description: Text("O MLX requer um iPhone físico com GPU Metal. Conecte um dispositivo e execute o app nele para usar o RAG com o modelo Qwen local.")
        )
    }

    private func dismissKeyboard() {
        isQuestionFocused = false
    }

    private func askQuestion() async {
        errorMessage = nil
        answer = nil

        guard let result = await engine.answer(question: question) else {
            if case .error(let message) = engine.state {
                errorMessage = message
            } else {
                errorMessage = "Não foi possível gerar uma resposta."
            }
            return
        }

        answer = result
    }
}

#Preview {
    RagView()
}
