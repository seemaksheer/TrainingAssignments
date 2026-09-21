#include <iostream>
#include <string>
#include <vector>

#include <windows.h>

#include "llama.h"

// ============================================================
// Configuration
// ============================================================

static const char * MODEL_PATH =
    "D:/LLM/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

static constexpr uint32_t CONTEXT_SIZE = 2048;
static constexpr int32_t GPU_LAYERS = 99;
static constexpr int MAX_GENERATION_TOKENS = 256;

// ============================================================
// llama.cpp log callback
// Display only errors.
// This suppresses normal CUDA/debug messages.
// ============================================================

void llama_log_callback(
    enum ggml_log_level level,
    const char * text,
    void * /*user_data*/)
{
    if (level >= GGML_LOG_LEVEL_ERROR)
    {
        std::cerr << text;
    }
}

// ============================================================
// Conversation history
// ============================================================

struct ChatHistory
{
    std::vector<std::string> roles;
    std::vector<std::string> contents;

    void add(
        const std::string & role,
        const std::string & content)
    {
        roles.push_back(role);
        contents.push_back(content);
    }

    size_t size() const
    {
        return roles.size();
    }
};

// ============================================================
// Apply model's default chat template
// ============================================================

bool apply_chat_template(
    const ChatHistory & history,
    std::string & formatted_prompt)
{
    std::vector<llama_chat_message> messages;

    messages.reserve(history.size());

    for (size_t i = 0; i < history.size(); ++i)
    {
        llama_chat_message message;

        message.role =
            history.roles[i].c_str();

        message.content =
            history.contents[i].c_str();

        messages.push_back(message);
    }

    // --------------------------------------------------------
    // First call determines required buffer size.
    // nullptr = use the model's default chat template.
    // --------------------------------------------------------

    int32_t required_size =
        llama_chat_apply_template(
            nullptr,
            messages.data(),
            messages.size(),
            true,
            nullptr,
            0);

    if (required_size <= 0)
    {
        std::cerr
            << "ERROR: Failed to determine chat-template size.\n";

        return false;
    }

    // --------------------------------------------------------
    // Allocate buffer.
    // --------------------------------------------------------

    std::vector<char> buffer(
        static_cast<size_t>(required_size) + 1,
        '\0');

    // --------------------------------------------------------
    // Apply chat template.
    // --------------------------------------------------------

    int32_t result =
        llama_chat_apply_template(
            nullptr,
            messages.data(),
            messages.size(),
            true,
            buffer.data(),
            static_cast<int32_t>(buffer.size()));

    if (result < 0)
    {
        std::cerr
            << "ERROR: Failed to apply chat template.\n";

        return false;
    }

    formatted_prompt.assign(
        buffer.data(),
        static_cast<size_t>(result));

    return true;
}

// ============================================================
// Tokenize text
// ============================================================

bool tokenize_text(
    const llama_vocab * vocab,
    const std::string & text,
    std::vector<llama_token> & tokens)
{
    tokens.clear();

    if (text.empty())
    {
        std::cerr
            << "ERROR: Prompt text is empty.\n";

        return false;
    }

    const int32_t text_length =
        static_cast<int32_t>(text.size());

    // --------------------------------------------------------
    // Determine required token count.
    // --------------------------------------------------------

    int32_t token_count =
        llama_tokenize(
            vocab,
            text.c_str(),
            text_length,
            nullptr,
            0,
            true,
            true);

    if (token_count < 0)
    {
        token_count = -token_count;
    }

    if (token_count <= 0)
    {
        std::cerr
            << "ERROR: Invalid token count: "
            << token_count
            << "\n";

        return false;
    }

    tokens.resize(
        static_cast<size_t>(token_count));

    // --------------------------------------------------------
    // Perform actual tokenization.
    // --------------------------------------------------------

    int32_t actual_count =
        llama_tokenize(
            vocab,
            text.c_str(),
            text_length,
            tokens.data(),
            token_count,
            true,
            true);

    if (actual_count < 0)
    {
        std::cerr
            << "ERROR: llama_tokenize failed. "
            << "Required tokens: "
            << -actual_count
            << "\n";

        tokens.clear();

        return false;
    }

    if (actual_count == 0)
    {
        std::cerr
            << "ERROR: llama_tokenize produced 0 tokens.\n";

        tokens.clear();

        return false;
    }

    tokens.resize(
        static_cast<size_t>(actual_count));

    return true;
}

// ============================================================
// Decode prompt tokens
// ============================================================

bool decode_tokens(
    llama_context * ctx,
    const std::vector<llama_token> & tokens)
{
    if (tokens.empty())
    {
        std::cerr
            << "ERROR: Cannot decode an empty token list.\n";

        return false;
    }

    // --------------------------------------------------------
    // Allocate batch.
    // --------------------------------------------------------

    llama_batch batch =
        llama_batch_init(
            static_cast<int32_t>(tokens.size()),
            0,
            1);

    // IMPORTANT:
    // llama_batch_init allocates the arrays but the actual
    // number of tokens must be explicitly assigned.
    batch.n_tokens =
        static_cast<int32_t>(tokens.size());

    // --------------------------------------------------------
    // Fill batch.
    // --------------------------------------------------------

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        batch.token[i] =
            tokens[i];

        batch.pos[i] =
            static_cast<llama_pos>(i);

        batch.n_seq_id[i] =
            1;

        batch.seq_id[i][0] =
            0;

        // Only the final prompt token needs logits.
        batch.logits[i] =
            (i == tokens.size() - 1);
    }

    // --------------------------------------------------------
    // Decode.
    // --------------------------------------------------------

    int32_t result =
        llama_decode(
            ctx,
            batch);

    llama_batch_free(batch);

    if (result != 0)
    {
        std::cerr
            << "ERROR: llama_decode failed. "
            << "Return code: "
            << result
            << "\n";

        return false;
    }

    return true;
}

// ============================================================
// Generate assistant response
// ============================================================

bool generate_response(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    std::string & response)
{
    response.clear();

    const uint32_t n_ctx =
        llama_n_ctx(ctx);

    // --------------------------------------------------------
    // Make sure there is room for generation.
    // --------------------------------------------------------

    if (prompt_tokens.size() +
            MAX_GENERATION_TOKENS >=
        n_ctx)
    {
        std::cerr
            << "ERROR: Prompt is too large for the "
               "configured context window.\n";

        return false;
    }

    // --------------------------------------------------------
    // Clear previous KV cache.
    //
    // The complete conversation is rebuilt and decoded
    // on every turn.
    // --------------------------------------------------------

    llama_memory_clear(
        llama_get_memory(ctx),
        true);

    // --------------------------------------------------------
    // Decode complete prompt.
    // --------------------------------------------------------

    if (!decode_tokens(
            ctx,
            prompt_tokens))
    {
        return false;
    }

    // ========================================================
    // Create sampler
    // ========================================================

    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();

    llama_sampler * sampler =
        llama_sampler_chain_init(
            sampler_params);

    if (sampler == nullptr)
    {
        std::cerr
            << "ERROR: Failed to initialize sampler.\n";

        return false;
    }

    // Temperature
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_temp(0.7f));

    // Top-P
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_top_p(0.9f, 1));

    // Random sampling
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_dist(1234));

    // ========================================================
    // Generate tokens
    // ========================================================

    llama_pos current_pos =
        static_cast<llama_pos>(
            prompt_tokens.size());

    for (int i = 0;
         i < MAX_GENERATION_TOKENS;
         ++i)
    {
        // ----------------------------------------------------
        // Sample next token.
        // ----------------------------------------------------

        llama_token token =
            llama_sampler_sample(
                sampler,
                ctx,
                -1);

        // ----------------------------------------------------
        // Stop when model reaches end-of-generation.
        // ----------------------------------------------------

        if (llama_vocab_is_eog(
                vocab,
                token))
        {
            break;
        }

        // ----------------------------------------------------
        // Accept token into sampler.
        // ----------------------------------------------------

        llama_sampler_accept(
            sampler,
            token);

        // ----------------------------------------------------
        // Convert token to text.
        // ----------------------------------------------------

        char piece[256];

        int32_t piece_size =
            llama_token_to_piece(
                vocab,
                token,
                piece,
                sizeof(piece),
                0,
                true);

        if (piece_size > 0)
        {
            std::string text(
                piece,
                static_cast<size_t>(
                    piece_size));

            response += text;

            std::cout
                << text
                << std::flush;
        }

        // ----------------------------------------------------
        // Decode generated token.
        // ----------------------------------------------------

        llama_batch batch =
            llama_batch_init(
                1,
                0,
                1);

        // IMPORTANT:
        // Explicitly set the number of tokens.
        batch.n_tokens = 1;

        batch.token[0] =
            token;

        batch.pos[0] =
            current_pos;

        batch.n_seq_id[0] =
            1;

        batch.seq_id[0][0] =
            0;

        batch.logits[0] =
            true;

        int32_t result =
            llama_decode(
                ctx,
                batch);

        llama_batch_free(batch);

        if (result != 0)
        {
            std::cerr
                << "\nERROR: llama_decode failed "
                   "during generation. "
                   "Return code: "
                << result
                << "\n";

            llama_sampler_free(
                sampler);

            return false;
        }

        ++current_pos;

        // ----------------------------------------------------
        // Context protection.
        // ----------------------------------------------------

        if (current_pos >=
            static_cast<llama_pos>(
                n_ctx - 1))
        {
            break;
        }
    }

    // --------------------------------------------------------
    // Free sampler.
    // --------------------------------------------------------

    llama_sampler_free(
        sampler);

    return true;
}

// ============================================================
// Main
// ============================================================

int main()
{
    // --------------------------------------------------------
    // Enable UTF-8 output in Windows console.
    // --------------------------------------------------------

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // --------------------------------------------------------
    // Suppress normal llama.cpp debug output.
    // Only errors are displayed.
    // --------------------------------------------------------

    llama_log_set(
        llama_log_callback,
        nullptr);

    std::cout
        << "=============================================\n"
        << "          MyLlamaApp - Qwen Chatbot\n"
        << "=============================================\n\n";

    std::cout
        << "Model : Qwen2.5-0.5B-Instruct Q4_K_M\n"
        << "GPU   : NVIDIA GeForce MX450\n"
        << "CUDA  : Enabled\n\n";

    // ========================================================
    // Initialize llama backend
    // ========================================================

    llama_backend_init();

    // ========================================================
    // Load model
    // ========================================================

    llama_model_params model_params =
        llama_model_default_params();

    // 99 = offload all available model layers.
    model_params.n_gpu_layers =
        GPU_LAYERS;

    std::cout
        << "Loading model...\n";

    llama_model * model =
        llama_model_load_from_file(
            MODEL_PATH,
            model_params);

    if (model == nullptr)
    {
        std::cerr
            << "ERROR: Could not load model:\n"
            << MODEL_PATH
            << "\n";

        llama_backend_free();

        return 1;
    }

    std::cout
        << "Model loaded successfully.\n";

    // ========================================================
    // Get vocabulary
    // ========================================================

    const llama_vocab * vocab =
        llama_model_get_vocab(model);

    if (vocab == nullptr)
    {
        std::cerr
            << "ERROR: Could not get model vocabulary.\n";

        llama_model_free(model);
        llama_backend_free();

        return 1;
    }

    std::cout
        << "Context size: "
        << CONTEXT_SIZE
        << "\n";

    std::cout
        << "GPU layers: "
        << GPU_LAYERS
        << "\n\n";

    // ========================================================
    // Create context
    // ========================================================

    llama_context_params context_params =
        llama_context_default_params();

    context_params.n_ctx =
        CONTEXT_SIZE;

    context_params.n_batch =
        CONTEXT_SIZE;

    context_params.n_ubatch =
        CONTEXT_SIZE;

    context_params.n_seq_max =
        1;

    llama_context * ctx =
        llama_init_from_model(
            model,
            context_params);

    if (ctx == nullptr)
    {
        std::cerr
            << "ERROR: Could not create llama context.\n";

        llama_model_free(model);
        llama_backend_free();

        return 1;
    }

    std::cout
        << "Context initialized successfully.\n\n";

    // ========================================================
    // Conversation history
    // ========================================================

    ChatHistory history;

    history.add(
        "system",
        "You are a helpful, concise and accurate AI "
        "assistant. Answer the user's questions directly. "
        "Do not invent facts. If you are unsure, say so.");

    // ========================================================
    // Chat interface
    // ========================================================

    std::cout
        << "Chat is ready.\n"
        << "Type 'exit' or 'quit' to close the application.\n\n";

    while (true)
    {
        // ----------------------------------------------------
        // User input
        // ----------------------------------------------------

        std::cout
            << "You: "
            << std::flush;

        std::string user_input;

        if (!std::getline(
                std::cin,
                user_input))
        {
            break;
        }

        // ----------------------------------------------------
        // Trim whitespace.
        // ----------------------------------------------------

        size_t start =
            user_input.find_first_not_of(
                " \t\r\n");

        size_t end =
            user_input.find_last_not_of(
                " \t\r\n");

        if (start == std::string::npos)
        {
            continue;
        }

        user_input =
            user_input.substr(
                start,
                end - start + 1);

        // ----------------------------------------------------
        // Exit commands.
        // ----------------------------------------------------

        if (user_input == "exit" ||
            user_input == "quit")
        {
            break;
        }

        // ----------------------------------------------------
        // Add user message.
        // ----------------------------------------------------

        history.add(
            "user",
            user_input);

        // ====================================================
        // Build formatted chat prompt.
        // ====================================================

        std::string formatted_prompt;

        if (!apply_chat_template(
                history,
                formatted_prompt))
        {
            std::cerr
                << "ERROR: Could not create chat prompt.\n";

            history.roles.pop_back();
            history.contents.pop_back();

            continue;
        }

        // ====================================================
        // Tokenize prompt.
        // ====================================================

        std::vector<llama_token> prompt_tokens;

        if (!tokenize_text(
                vocab,
                formatted_prompt,
                prompt_tokens))
        {
            std::cerr
                << "ERROR: Could not tokenize prompt.\n";

            history.roles.pop_back();
            history.contents.pop_back();

            continue;
        }

        // ----------------------------------------------------
        // Diagnostic information.
        // ----------------------------------------------------

        std::cout
            << "[Prompt tokens: "
            << prompt_tokens.size()
            << "]\n";

        // ====================================================
        // Context management
        // ====================================================

        while (prompt_tokens.size() +
                   MAX_GENERATION_TOKENS >=
               CONTEXT_SIZE)
        {
            // ------------------------------------------------
            // Keep system message.
            // Remove oldest user/assistant pair.
            // ------------------------------------------------

            if (history.size() <= 3)
            {
                std::cerr
                    << "ERROR: Conversation is too large "
                       "for the 2048-token context.\n";

                history.roles.pop_back();
                history.contents.pop_back();

                break;
            }

            // Remove oldest user message.
            history.roles.erase(
                history.roles.begin() + 1);

            history.contents.erase(
                history.contents.begin() + 1);

            // Remove corresponding assistant message.
            history.roles.erase(
                history.roles.begin() + 1);

            history.contents.erase(
                history.contents.begin() + 1);

            // Rebuild formatted prompt.
            if (!apply_chat_template(
                    history,
                    formatted_prompt))
            {
                std::cerr
                    << "ERROR: Could not rebuild chat prompt.\n";

                break;
            }

            // Retokenize.
            if (!tokenize_text(
                    vocab,
                    formatted_prompt,
                    prompt_tokens))
            {
                std::cerr
                    << "ERROR: Could not tokenize rebuilt prompt.\n";

                break;
            }
        }

        // ----------------------------------------------------
        // Final context check.
        // ----------------------------------------------------

        if (prompt_tokens.empty())
        {
            std::cerr
                << "ERROR: Prompt contains no tokens.\n";

            continue;
        }

        if (prompt_tokens.size() +
                MAX_GENERATION_TOKENS >=
            CONTEXT_SIZE)
        {
            std::cerr
                << "ERROR: Prompt still exceeds "
                   "the context window.\n";

            continue;
        }

        // ====================================================
        // Generate response.
        // ====================================================

        std::cout
            << "\nAI: "
            << std::flush;

        std::string assistant_response;

        bool success =
            generate_response(
                ctx,
                vocab,
                prompt_tokens,
                assistant_response);

        std::cout
            << "\n\n";

        if (!success)
        {
            std::cerr
                << "Generation failed.\n";

            // Remove failed user message.
            history.roles.pop_back();
            history.contents.pop_back();

            continue;
        }

        // ====================================================
        // Save assistant response.
        //
        // This allows subsequent questions to use the
        // previous conversation.
        // ====================================================

        history.add(
            "assistant",
            assistant_response);
    }

    // ========================================================
    // Cleanup
    // ========================================================

    std::cout
        << "\nShutting down...\n";

    // Your llama.cpp version uses llama_free().
    llama_free(ctx);

    llama_model_free(model);

    llama_backend_free();

    std::cout
        << "Goodbye!\n";

    return 0;
}