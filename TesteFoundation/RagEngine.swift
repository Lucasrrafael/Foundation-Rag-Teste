//
//  RagEngine.swift
//  TesteFoundation
//

import Foundation
import FoundationModels
import NaturalLanguage
import PDFKit

struct RagPassage: Identifiable, Sendable {
    let id: String
    let sourceTitle: String
    let text: String
}

struct RagSource: Identifiable, Sendable {
    let id: String
    let sourceTitle: String
    let excerpt: String
    let score: Double
}

struct RagAnswer: Sendable {
    let response: String
    let sources: [RagSource]
    let retrievalMode: RagRetrievalMode
}

enum RagRetrievalMode: String, Sendable {
    case embedding = "Busca semântica (NLEmbedding)"
    case keyword = "Busca por palavra-chave"
}

enum RagEngineState: Equatable, Sendable {
    case idle
    case indexing
    case ready(passageCount: Int, mode: RagRetrievalMode)
    case answering
    case error(String)
}

@Observable
@MainActor
final class RagEngine {
    private struct IndexedPassage {
        let passage: RagPassage
        let vector: [Double]?
    }

    private struct BundledPDF {
        let resourceName: String
        let title: String
    }

    private let pdfs: [BundledPDF] = [
        BundledPDF(resourceName: "cartilha_maio_laranja", title: "Cartilha Maio Laranja 2021"),
        BundledPDF(resourceName: "cartilha_ists", title: "Cartilha ISTs - Mitos e Verdades"),
        BundledPDF(resourceName: "cartilha_ist_prevencao", title: "Cartilha IST - Prevenção e Sexualidade")
    ]

    private let chunkSize = 750
    private let topK = 4

    private(set) var state: RagEngineState = .idle
    private var indexedPassages: [IndexedPassage] = []
    private var retrievalMode: RagRetrievalMode = .keyword
    private var embedding: NLEmbedding?

    func indexDocuments() async {
        state = .indexing
        indexedPassages = []

        do {
            embedding = NLEmbedding.sentenceEmbedding(for: .portuguese)
                ?? NLEmbedding.sentenceEmbedding(for: .undetermined)
            retrievalMode = embedding != nil ? .embedding : .keyword

            var allPassages: [RagPassage] = []

            for pdf in pdfs {
                guard let url = Bundle.main.url(forResource: pdf.resourceName, withExtension: "pdf") else {
                    throw RagEngineError.missingPDF(pdf.resourceName)
                }

                let text = try extractText(from: url)
                let chunks = chunk(text: text, sourceTitle: pdf.title)
                allPassages.append(contentsOf: chunks)
            }

            guard !allPassages.isEmpty else {
                throw RagEngineError.emptyIndex
            }

            indexedPassages = allPassages.map { passage in
                let vector = embedding?.vector(for: passage.text)
                return IndexedPassage(passage: passage, vector: vector)
            }

            state = .ready(passageCount: indexedPassages.count, mode: retrievalMode)
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    func answer(question: String) async -> RagAnswer? {
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuestion.isEmpty else { return nil }

        guard case .ready = state else { return nil }

        state = .answering

        let sources = topK(for: trimmedQuestion)
        let prompt = buildPrompt(question: trimmedQuestion, sources: sources)

        let session = LanguageModelSession(
            instructions: """
            Você é um assistente educativo de saúde e prevenção. O contexto fornecido vem de \
            cartilhas oficiais de caráter educativo, informativo e de prevenção, produzidas por \
            órgãos públicos e instituições de ensino para conscientização e proteção da população.
            O objetivo é estritamente educacional e preventivo, ajudando pessoas a se informarem, \
            se protegerem e buscarem ajuda.

            Responda às perguntas usando apenas o contexto fornecido pelas cartilhas.
            Responda em português, de forma clara, objetiva, respeitosa e apropriada, mantendo um \
            tom informativo e profissional de saúde pública.
            Se a resposta não estiver no contexto, diga explicitamente que não encontrou essa \
            informação nas cartilhas.
            Não invente informações.
            """
        )

        do {
            let result = try await session.respond(to: prompt)
            state = .ready(passageCount: indexedPassages.count, mode: retrievalMode)
            return RagAnswer(
                response: result.content,
                sources: sources,
                retrievalMode: retrievalMode
            )
        } catch {
            state = .error(error.localizedDescription)
            return nil
        }
    }

    // MARK: - PDF extraction

    private func extractText(from url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw RagEngineError.unreadablePDF(url.lastPathComponent)
        }

        var pages: [String] = []
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index), let pageText = page.string else { continue }
            let cleaned = pageText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty {
                pages.append(cleaned)
            }
        }

        let fullText = pages.joined(separator: "\n\n")
        guard !fullText.isEmpty else {
            throw RagEngineError.noTextInPDF(url.lastPathComponent)
        }

        return fullText
    }

    // MARK: - Chunking

    private func chunk(text: String, sourceTitle: String) -> [RagPassage] {
        let paragraphs = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 40 }

        var chunks: [String] = []
        var current = ""

        for paragraph in paragraphs {
            if current.isEmpty {
                current = paragraph
            } else if current.count + paragraph.count + 2 <= chunkSize {
                current += "\n\n" + paragraph
            } else {
                chunks.append(current)
                current = paragraph
            }
        }

        if !current.isEmpty {
            chunks.append(current)
        }

        return chunks.enumerated().map { index, chunkText in
            RagPassage(
                id: "\(sourceTitle)-\(index)",
                sourceTitle: sourceTitle,
                text: chunkText
            )
        }
    }

    // MARK: - Retrieval

    private func topK(for question: String) -> [RagSource] {
        if retrievalMode == .embedding,
           let embedding,
           let queryVector = embedding.vector(for: question) {
            return rankedByEmbedding(queryVector: queryVector)
        }

        return rankedByKeywords(question: question)
    }

    private func rankedByEmbedding(queryVector: [Double]) -> [RagSource] {
        let scored = indexedPassages.compactMap { indexed -> RagSource? in
            guard let vector = indexed.vector else { return nil }
            let score = cosineSimilarity(queryVector, vector)
            return RagSource(
                id: indexed.passage.id,
                sourceTitle: indexed.passage.sourceTitle,
                excerpt: excerpt(from: indexed.passage.text),
                score: score
            )
        }

        return Array(scored.sorted { $0.score > $1.score }.prefix(topK))
    }

    private func rankedByKeywords(question: String) -> [RagSource] {
        let queryTerms = tokenize(question)
        guard !queryTerms.isEmpty else { return [] }

        let scored = indexedPassages.map { indexed in
            let passageTerms = Set(tokenize(indexed.passage.text))
            let overlap = queryTerms.filter { passageTerms.contains($0) }.count
            let normalizedScore = Double(overlap) / Double(queryTerms.count)

            return RagSource(
                id: indexed.passage.id,
                sourceTitle: indexed.passage.sourceTitle,
                excerpt: excerpt(from: indexed.passage.text),
                score: normalizedScore
            )
        }

        return Array(scored.sorted { $0.score > $1.score }.prefix(topK))
    }

    private func buildPrompt(question: String, sources: [RagSource]) -> String {
        let context = sources.enumerated().map { index, source in
            """
            [Fonte \(index + 1): \(source.sourceTitle)]
            \(source.excerpt)
            """
        }.joined(separator: "\n\n")

        return """
        Contexto das cartilhas:
        \(context)

        Pergunta: \(question)
        """
    }

    // MARK: - Helpers

    private func excerpt(from text: String, limit: Int = 320) -> String {
        if text.count <= limit { return text }
        return String(text.prefix(limit)) + "…"
    }

    private func tokenize(_ text: String) -> [String] {
        text
            .lowercased()
            .folding(options: .diacriticInsensitive, locale: .current)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
    }

    private func cosineSimilarity(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }

        var dot = 0.0
        var lhsNorm = 0.0
        var rhsNorm = 0.0

        for index in lhs.indices {
            dot += lhs[index] * rhs[index]
            lhsNorm += lhs[index] * lhs[index]
            rhsNorm += rhs[index] * rhs[index]
        }

        let denominator = sqrt(lhsNorm) * sqrt(rhsNorm)
        guard denominator > 0 else { return 0 }
        return dot / denominator
    }
}

enum RagEngineError: LocalizedError {
    case missingPDF(String)
    case unreadablePDF(String)
    case noTextInPDF(String)
    case emptyIndex

    var errorDescription: String? {
        switch self {
        case .missingPDF(let name):
            return "PDF não encontrado no bundle: \(name).pdf"
        case .unreadablePDF(let name):
            return "Não foi possível abrir o PDF: \(name)"
        case .noTextInPDF(let name):
            return "Nenhum texto extraível no PDF: \(name)"
        case .emptyIndex:
            return "Nenhuma passagem foi indexada a partir dos PDFs."
        }
    }
}
