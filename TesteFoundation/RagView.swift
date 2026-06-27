//
//  RagView.swift
//  TesteFoundation
//

import SwiftUI
import FoundationModels

struct RagView: View {
    private let model = SystemLanguageModel.default

    @State private var engine = RagEngine()
    @FocusState private var isQuestionFocused: Bool

    @State private var question = "O que é abuso sexual contra crianças?"
    @State private var answer: RagAnswer?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Group {
                switch model.availability {
                case .available:
                    mainContent
                case .unavailable(.appleIntelligenceNotEnabled):
                    unavailableView(
                        title: "Apple Intelligence desativado",
                        message: "Ative o Apple Intelligence em Ajustes para usar o RAG com o modelo local."
                    )
                case .unavailable(.deviceNotEligible):
                    unavailableView(
                        title: "Dispositivo não suportado",
                        message: "Este dispositivo não é compatível com Apple Intelligence."
                    )
                case .unavailable(.modelNotReady):
                    unavailableView(
                        title: "Modelo não pronto",
                        message: "O modelo ainda está sendo preparado. Tente novamente em instantes."
                    )
                case .unavailable:
                    unavailableView(
                        title: "Modelo indisponível",
                        message: "O Foundation Model não está disponível no momento."
                    )
                }
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
                        Task { await engine.indexDocuments() }
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

    private func unavailableView(title: String, message: String) -> some View {
        ContentUnavailableView(title, systemImage: "apple.intelligence", description: Text(message))
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
