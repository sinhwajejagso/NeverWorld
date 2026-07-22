package com.neverworld.single

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.room.*
import coil.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

// ============================================================================
// 1. 데이터베이스 및 엔티티 (Room DB & 로컬 벡터 Search 최적화)
// ============================================================================

/**
 * FloatArray <-> ByteArray 간 고속 직렬화를 위한 Room TypeConverter
 */
class VectorConverters {
    @TypeConverter
    fun fromFloatArray(array: FloatArray?): ByteArray? {
        if (array == null || array.isEmpty()) return null
        val buffer = ByteBuffer.allocate(array.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        val floatBuffer = buffer.asFloatBuffer()
        floatBuffer.put(array)
        return buffer.array()
    }

    @TypeConverter
    fun toFloatArray(bytes: ByteArray?): FloatArray? {
        if (bytes == null || bytes.size < 4) return null
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floatBuffer = buffer.asFloatBuffer()
        val array = FloatArray(floatBuffer.remaining())
        floatBuffer.get(array)
        return array
    }
}

/**
 * 로어북(세계관 설정 및 기억) 엔티티
 */
@Entity(tableName = "lorebook")
@TypeConverters(VectorConverters::class)
data class LorebookEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val characterId: String,
    val keyword: String,
    val content: String,
    val embeddingVector: FloatArray? = null
)

/**
 * 대화 내역 엔티티
 */
@Entity(tableName = "chat_history")
data class ChatMessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val characterId: String,
    val sender: String, // "user" 또는 "assistant"
    val text: String,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * 캐릭터 설정 엔티티
 */
@Entity(tableName = "character_persona")
data class CharacterPersonaEntity(
    @PrimaryKey val characterId: String,
    val name: String,
    val description: String,
    val personality: String,
    val scenario: String,
    val firstMes: String,
    val systemPrompt: String
)

@Dao
interface LorebookDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLorebook(entry: LorebookEntry): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLorebooks(entries: List<LorebookEntry>)

    @Query("SELECT * FROM lorebook WHERE characterId = :characterId")
    suspend fun getLorebookByCharacter(characterId: String): List<LorebookEntry>

    @Query("DELETE FROM lorebook WHERE characterId = :characterId")
    suspend fun deleteLorebookByCharacter(characterId: String)
}

@Dao
interface ChatDao {
    @Insert
    suspend fun insertMessage(message: ChatMessageEntity): Long

    @Query("SELECT * FROM chat_history WHERE characterId = :characterId ORDER BY timestamp ASC")
    suspend fun getChatHistory(characterId: String): List<ChatMessageEntity>

    @Query("DELETE FROM chat_history WHERE characterId = :characterId")
    suspend fun clearChatHistory(characterId: String)
}

@Dao
interface CharacterDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCharacter(character: CharacterPersonaEntity)

    @Query("SELECT * FROM character_persona WHERE characterId = :characterId")
    suspend fun getCharacter(characterId: String): CharacterPersonaEntity?

    @Query("SELECT * FROM character_persona")
    suspend fun getAllCharacters(): List<CharacterPersonaEntity>
}

@Database(
    entities = [LorebookEntry::class, ChatMessageEntity::class, CharacterPersonaEntity::class],
    version = 1,
    exportSchema = false
)
@TypeConverters(VectorConverters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun lorebookDao(): LorebookDao
    abstract fun chatDao(): ChatDao
    abstract fun characterDao(): CharacterDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "neverworld_single.db"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}

/**
 * 코사인 유사도 연산 유틸리티 (수학 수식 정밀 계산)
 */
object VectorMath {
    fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size || v1.isEmpty()) return 0f
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        for (i in v1.indices) {
            val a = v1[i]
            val b = v2[i]
            dotProduct += a * b
            normA += a * a
            normB += b * b
        }
        val denominator = sqrt(normA) * sqrt(normB)
        return if (denominator == 0f) 0f else dotProduct / denominator
    }
}

// ============================================================================
// 2. 네트워크 API 레포지토리 (Grok SSE 스트리밍 타임아웃 예외 처리 완비)
// ============================================================================

class ApiRepository {

    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    // 일반 REST 요청용 Client
    private val restClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    // SSE 스트리밍 전용 Client (readTimeout = 0 필수: 무한 대기 허용)
    private val sseClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    /**
     * 외부 API를 호출하여 텍스트를 float 벡터 배열로 변환
     */
    suspend fun fetchEmbedding(apiKey: String, text: String): FloatArray? = withContext(Dispatchers.IO) {
        if (apiKey.isBlank() || text.isBlank()) return@withContext null
        try {
            val jsonPayload = JSONObject().apply {
                put("input", text)
                put("model", "text-embedding-3-small")
            }
            val request = Request.Builder()
                .url("https://api.openai.com/v1/embeddings")
                .addHeader("Authorization", "Bearer $apiKey")
                .post(jsonPayload.toString().toRequestBody(jsonMediaType))
                .build()

            restClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext null
                val bodyStr = response.body?.string() ?: return@withContext null
                val jsonRes = JSONObject(bodyStr)
                val embeddingArray = jsonRes.getJSONArray("data")
                    .getJSONObject(0)
                    .getJSONArray("embedding")

                val floatArray = FloatArray(embeddingArray.length())
                for (i in 0 until embeddingArray.length()) {
                    floatArray[i] = embeddingArray.getDouble(i).toFloat()
                }
                floatArray
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * xAI Grok 대화 completion API SSE 스트리밍 호출 (EventSource 리소스 취소 기능 포함)
     */
    fun streamGrokChat(
        apiKey: String,
        messages: List<JSONObject>,
        onTokenReceived: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (Throwable) -> Unit
    ): EventSource {
        val jsonPayload = JSONObject().apply {
            put("model", "grok-4")
            put("messages", JSONArray(messages))
            put("stream", true)
            put("temperature", 0.85)
        }

        val request = Request.Builder()
            .url("https://api.x.ai/v1/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Accept", "text/event-stream")
            .post(jsonPayload.toString().toRequestBody(jsonMediaType))
            .build()

        val factory = EventSources.createFactory(sseClient)
        val eventSource = factory.newEventSource(request, object : EventSourceListener() {
            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data == "[DONE]") {
                    onComplete()
                    return
                }
                try {
                    val json = JSONObject(data)
                    val choices = json.optJSONArray("choices") ?: return
                    if (choices.length() > 0) {
                        val delta = choices.getJSONObject(0).optJSONObject("delta")
                        val content = delta?.optString("content", "") ?: ""
                        if (content.isNotEmpty()) {
                            onTokenReceived(content)
                        }
                    }
                } catch (e: Exception) {
                    // 통신 파싱 에러 방지
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                onError(t ?: Exception("서버 통신 중 연결이 끊어졌거나 오류가 발생했습니다."))
            }
        })
        return eventSource
    }

    /**
     * xAI Grok 이미지 생성 API 호출
     */
    suspend fun generateSceneImage(apiKey: String, prompt: String): String? = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) return@withContext null
        try {
            val jsonPayload = JSONObject().apply {
                put("prompt", "Masterpiece dark visual novel anime illustration, detailed: $prompt")
                put("model", "grok-image-1")
                put("n", 1)
            }

            val request = Request.Builder()
                .url("https://api.x.ai/v1/images/generations")
                .addHeader("Authorization", "Bearer $apiKey")
                .post(jsonPayload.toString().toRequestBody(jsonMediaType))
                .build()

            restClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext null
                val bodyStr = response.body?.string() ?: return@withContext null
                val jsonRes = JSONObject(bodyStr)
                val dataArray = jsonRes.optJSONArray("data")
                if (dataArray != null && dataArray.length() > 0) {
                    dataArray.getJSONObject(0).optString("url", null)
                } else null
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

// ============================================================================
// 3. ViewModel & 상태 관리 (스레드 안전성 확보)
// ============================================================================

data class ChatUiMessage(
    val id: Long = 0,
    val sender: String,
    val text: String
)

data class UiState(
    val characterId: String = "default_char",
    val characterName: String = "캐릭터 선택 안됨",
    val characterList: List<CharacterPersonaEntity> = emptyList(),
    val messages: List<ChatUiMessage> = emptyList(),
    val isStreaming: Boolean = false,
    val backgroundImageUrl: String? = null,
    val isClimaxOrIntense: Boolean = false,
    val currentRAGContext: String = "",
    val openAiApiKey: String = "",
    val grokApiKey: String = "",
    val showSettingsDialog: Boolean = false,
    val statusMessage: String? = null
)

class ChatViewModel(
    private val db: AppDatabase,
    private val repository: ApiRepository,
    private val sharedPrefs: SharedPreferences
) : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var activeEventSource: EventSource? = null
    private var streamingJob: Job? = null

    init {
        val openAiKey = sharedPrefs.getString("openai_key", "") ?: ""
        val grokKey = sharedPrefs.getString("grok_key", "") ?: ""
        _uiState.update { it.copy(openAiApiKey = openAiKey, grokApiKey = grokKey) }
        
        loadCharactersAndSelectDefault()
    }

    /**
     * API 키 설정 및 저장
     */
    fun updateApiKeys(openAiKey: String, grokKey: String) {
        sharedPrefs.edit()
            .putString("openai_key", openAiKey)
            .putString("grok_key", grokKey)
            .apply()

        _uiState.update {
            it.copy(
                openAiApiKey = openAiKey,
                grokApiKey = grokKey,
                showSettingsDialog = false,
                statusMessage = "API 키가 성공적으로 저장되었습니다."
            )
        }
    }

    fun toggleSettingsDialog(show: Boolean) {
        _uiState.update { it.copy(showSettingsDialog = show) }
    }

    /**
     * 캐릭터 목록 및 기본 대화 내역 로딩
     */
    fun loadCharactersAndSelectDefault(targetCharId: String? = null) {
        viewModelScope.launch {
            val allChars = db.characterDao().getAllCharacters()
            if (allChars.isEmpty()) {
                val defaultChar = CharacterPersonaEntity(
                    characterId = "asuka_default",
                    name = "아스카",
                    description = "당신에게 기묘할 정도로 집착하는 인물.",
                    personality = "차가운 언변 뒤에 숨겨진 잔혹함과 매혹적인 분위기.",
                    scenario = "어두운 저택의 다이닝 룸에서 둘만의 대화를 나누는 상황.",
                    firstMes = "*당신을 오랫동안 침묵 속에서 가만히 바라보며* 「결국... 다시 내게로 돌아왔군.」",
                    systemPrompt = "당신은 어두운 비주얼 노벨의 주인공 '아스카'입니다. 지문은 「」 또는 * *로 감싸고 대사는 \" \"로 구분하세요."
                )
                db.characterDao().insertCharacter(defaultChar)
                db.chatDao().insertMessage(ChatMessageEntity(characterId = defaultChar.characterId, sender = "assistant", text = defaultChar.firstMes))
                loadCharacterData(defaultChar.characterId, listOf(defaultChar))
            } else {
                val charToSelect = targetCharId ?: allChars.first().characterId
                loadCharacterData(charToSelect, allChars)
            }
        }
    }

    private suspend fun loadCharacterData(characterId: String, charList: List<CharacterPersonaEntity>) {
        val character = db.characterDao().getCharacter(characterId) ?: return
        val history = db.chatDao().getChatHistory(characterId)
        val uiMsgs = history.map { ChatUiMessage(it.id, it.sender, it.text) }

        _uiState.update {
            it.copy(
                characterId = character.characterId,
                characterName = character.name,
                characterList = charList,
                messages = uiMsgs,
                currentRAGContext = ""
            )
        }
    }

    /**
     * 대화 내역 초기화
     */
    fun clearHistory() {
        viewModelScope.launch {
            db.chatDao().clearChatHistory(_uiState.value.characterId)
            val character = db.characterDao().getCharacter(_uiState.value.characterId)
            if (character != null) {
                db.chatDao().insertMessage(ChatMessageEntity(characterId = character.characterId, sender = "assistant", text = character.firstMes))
            }
            loadCharacterData(_uiState.value.characterId, _uiState.value.characterList)
        }
    }

    /**
     * 메시지 전송 및 RAG 조립 후 Grok API 스트리밍 답변 실행
     */
    fun sendMessage(userText: String) {
        if (userText.isBlank() || _uiState.value.isStreaming) return

        val grokKey = _uiState.value.grokApiKey
        if (grokKey.isBlank()) {
            _uiState.update { it.copy(showSettingsDialog = true, statusMessage = "xAI Grok API 키를 먼저 입력해주세요.") }
            return
        }

        streamingJob = viewModelScope.launch {
            val charId = _uiState.value.characterId

            // 1. 유저 메시지 DB 저장 및 UI 반영
            val userEntity = ChatMessageEntity(characterId = charId, sender = "user", text = userText)
            db.chatDao().insertMessage(userEntity)

            val updatedMessages = _uiState.value.messages + ChatUiMessage(sender = "user", text = userText)
            _uiState.update { it.copy(messages = updatedMessages, isStreaming = true) }

            // 2. RAG (OpenAI 임베딩 API 기반 검색)
            var ragPromptInsert = ""
            val openAiKey = _uiState.value.openAiApiKey
            if (openAiKey.isNotBlank()) {
                val userVector = repository.fetchEmbedding(openAiKey, userText)
                if (userVector != null) {
                    val lorebooks = db.lorebookDao().getLorebookByCharacter(charId)
                    val matchedEntries = lorebooks.mapNotNull { entry ->
                        entry.embeddingVector?.let { vector ->
                            val sim = VectorMath.cosineSimilarity(userVector, vector)
                            if (sim > 0.35f) Pair(entry, sim) else null
                        }
                    }.sortedByDescending { it.second }.take(3)

                    if (matchedEntries.isNotEmpty()) {
                        ragPromptInsert = matchedEntries.joinToString("\n") { "[기억: ${it.first.keyword}] ${it.first.content}" }
                        _uiState.update { it.copy(currentRAGContext = "RAG 매칭 완료: ${matchedEntries.size}건 메모리 반영됨") }
                    }
                }
            }

            // 3. API 요청용 프롬프트 데이터 구성
            val character = db.characterDao().getCharacter(charId)
            val apiMessages = mutableListOf<JSONObject>()

            val systemContent = buildString {
                append(character?.systemPrompt ?: "")
                append("\n[캐릭터 페르소나]\n").append(character?.description ?: "")
                if (!ragPromptInsert.isBlank()) {
                    append("\n\n[장기 기억 데이터베이스 (RAG)]\n").append(ragPromptInsert)
                }
            }

            apiMessages.add(JSONObject().apply {
                put("role", "system")
                put("content", systemContent)
            })

            val history = db.chatDao().getChatHistory(charId).takeLast(12)
            for (msg in history) {
                apiMessages.add(JSONObject().apply {
                    put("role", if (msg.sender == "user") "user" else "assistant")
                    put("content", msg.text)
                })
            }

            // 4. UI 렌더링용 어시스턴트 메시지 자리 확보
            var currentAiResponse = ""
            val assistantIndex = updatedMessages.size
            val listWithPlaceholder = updatedMessages + ChatUiMessage(sender = "assistant", text = "")
            _uiState.update { it.copy(messages = listWithPlaceholder) }

            // 5. SSE 스트리밍 통신 시작
            activeEventSource = repository.streamGrokChat(
                apiKey = grokKey,
                messages = apiMessages,
                onTokenReceived = { token ->
                    currentAiResponse += token
                    _uiState.update { state ->
                        val newList = state.messages.toMutableList()
                        if (assistantIndex < newList.size) {
                            newList[assistantIndex] = ChatUiMessage(sender = "assistant", text = currentAiResponse)
                        }
                        
                        val isClimax = currentAiResponse.contains("분노") || currentAiResponse.contains("피") ||
                                       currentAiResponse.contains("집착") || currentAiResponse.contains("절망") ||
                                       currentAiResponse.contains("클라이맥스")

                        state.copy(messages = newList, isClimaxOrIntense = isClimax)
                    }
                },
                onComplete = {
                    viewModelScope.launch {
                        db.chatDao().insertMessage(
                            ChatMessageEntity(characterId = charId, sender = "assistant", text = currentAiResponse)
                        )
                        _uiState.update { it.copy(isStreaming = false) }

                        // 묘사 문맥 감지 시 Grok 이미지 자동 생성
                        if (grokKey.isNotBlank() && (currentAiResponse.contains("장면") || currentAiResponse.length > 180)) {
                            generateBackgroundScene(grokKey, currentAiResponse)
                        }
                    }
                },
                onError = { throwable ->
                    viewModelScope.launch {
                        _uiState.update {
                            it.copy(
                                isStreaming = false,
                                statusMessage = "오류 발생: ${throwable.localizedMessage}"
                            )
                        }
                    }
                }
            )
        }
    }

    private fun generateBackgroundScene(apiKey: String, sceneDescription: String) {
        viewModelScope.launch {
            val imageUrl = repository.generateSceneImage(apiKey, sceneDescription)
            if (imageUrl != null) {
                _uiState.update { it.copy(backgroundImageUrl = imageUrl) }
            }
        }
    }

    /**
     * SillyTavern JSON 캐릭터 카드를 파싱하여 캐릭터 정보 및 로어북적재
     */
    fun importSillyTavernJson(jsonString: String) {
        viewModelScope.launch {
            try {
                val rootObj = JSONObject(jsonString)
                // SillyTavern V1 / V2 데이터 호환 처리
                val dataObj = rootObj.optJSONObject("data") ?: rootObj

                val name = dataObj.optString("name", "신규 캐릭터")
                val description = dataObj.optString("description", "")
                val personality = dataObj.optString("personality", "")
                val scenario = dataObj.optString("scenario", "")
                val firstMes = dataObj.optString("first_mes", "안녕하세요.")
                val newCharId = "char_${System.currentTimeMillis()}"

                val newChar = CharacterPersonaEntity(
                    characterId = newCharId,
                    name = name,
                    description = description,
                    personality = personality,
                    scenario = scenario,
                    firstMes = firstMes,
                    systemPrompt = "캐릭터 이름: $name\n성격: $personality\n설명: $description"
                )
                db.characterDao().insertCharacter(newChar)
                db.chatDao().insertMessage(ChatMessageEntity(characterId = newCharId, sender = "assistant", text = firstMes))

                // 로어북(Character Book) 항목 파싱
                val characterBook = dataObj.optJSONObject("character_book")
                if (characterBook != null) {
                    val entries = characterBook.optJSONArray("entries")
                    if (entries != null) {
                        val lorebookList = mutableListOf<LorebookEntry>()
                        for (i in 0 until entries.length()) {
                            val entryObj = entries.getJSONObject(i)
                            val keysArray = entryObj.optJSONArray("keys")
                            val keyword = if (keysArray != null && keysArray.length() > 0) keysArray.getString(0) else "설정_$i"
                            val content = entryObj.optString("content", "")

                            if (content.isNotBlank()) {
                                // OpenAI API 키가 설정되어 있으면 벡터 임베딩 생성
                                val openAiKey = _uiState.value.openAiApiKey
                                val vector = if (openAiKey.isNotBlank()) repository.fetchEmbedding(openAiKey, "$keyword $content") else null
                                lorebookList.add(
                                    LorebookEntry(
                                        characterId = newCharId,
                                        keyword = keyword,
                                        content = content,
                                        embeddingVector = vector
                                    )
                                )
                            }
                        }
                        if (lorebookList.isNotEmpty()) {
                            db.lorebookDao().insertLorebooks(lorebookList)
                        }
                    }
                }

                loadCharactersAndSelectDefault(newCharId)
                _uiState.update { it.copy(statusMessage = "'$name' 캐릭터 카드가 정상적으로 성공 내보내기/가져오기 되었습니다.") }

            } catch (e: Exception) {
                _uiState.update { it.copy(statusMessage = "JSON 파싱 실패: 잘못된 캐릭터 카드 규격입니다.") }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        activeEventSource?.cancel()
        streamingJob?.cancel()
    }
}

class ChatViewModelFactory(
    private val db: AppDatabase,
    private val repository: ApiRepository,
    private val sharedPrefs: SharedPreferences
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ChatViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return ChatViewModel(db, repository, sharedPrefs) as T
        }
        throw IllegalArgumentException("알 수 없는 ViewModel 클래스입니다.")
    }
}

// ============================================================================
// 4. Jetpack Compose Visual Novel UI (수정 및 UI 퍼포먼스 최적화)
// ============================================================================

class MainActivity : ComponentActivity() {

    private val db by lazy { AppDatabase.getDatabase(this) }
    private val repository by lazy { ApiRepository() }
    private val sharedPrefs by lazy { getSharedPreferences("neverworld_prefs", Context.MODE_PRIVATE) }

    private val viewModel: ChatViewModel by viewModels {
        ChatViewModelFactory(db, repository, sharedPrefs)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            NeverWorldTheme {
                val uiState by viewModel.uiState.collectAsState()
                ChatScreen(
                    uiState = uiState,
                    onSendMessage = { text -> viewModel.sendMessage(text) },
                    onSelectCharacter = { id -> viewModel.loadCharactersAndSelectDefault(id) },
                    onClearHistory = { viewModel.clearHistory() },
                    onSaveApiKeys = { openAi, grok -> viewModel.updateApiKeys(openAi, grok) },
                    onToggleSettings = { show -> viewModel.toggleSettingsDialog(show) },
                    onImportJson = { json -> viewModel.importSillyTavernJson(json) }
                )
            }
        }
    }
}

@Composable
fun NeverWorldTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF8B0000), // Deep Burgundy
            background = Color(0xFF0F0F0F),
            surface = Color(0xFF1E1E1E)
        ),
        content = content
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    uiState: UiState,
    onSendMessage: (String) -> Unit,
    onSelectCharacter: (String) -> Unit,
    onClearHistory: () -> Unit,
    onSaveApiKeys: (String, String) -> Unit,
    onToggleSettings: (Boolean) -> Unit,
    onImportJson: (String) -> Unit
) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val context = LocalContext.current

    // JSON 파일 임포터 Launcher
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                context.contentResolver.openInputStream(it)?.use { stream ->
                    val reader = BufferedReader(InputStreamReader(stream))
                    val jsonStr = reader.readText()
                    onImportJson(jsonStr)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    // 버건디 네온 테두리 애니메이션
    val infiniteTransition = rememberInfiniteTransition(label = "NeonTransition")
    val neonAlpha by infiniteTransition.animateFloat(
        initialValue = 0.2f,
        targetValue = if (uiState.isClimaxOrIntense) 1.0f else 0.25f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "NeonAlpha"
    )

    // 신규 메세지 수신 시 자동 하단 스크롤
    LaunchedEffect(uiState.messages.size, uiState.messages.lastOrNull()?.text) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .border(
                width = if (uiState.isClimaxOrIntense) 4.dp else 1.5.dp,
                color = Color(0xFF8B0000).copy(alpha = neonAlpha)
            )
    ) {
        // 백그라운드 레이어: Grok 실시간 비주얼 노벨 일러스트 (Crossfade 애니메이션)
        Crossfade(targetState = uiState.backgroundImageUrl, label = "BgCrossfade") { url ->
            if (url != null) {
                AsyncImage(
                    model = url,
                    contentDescription = "장면 일러스트",
                    modifier = Modifier
                        .fillMaxSize()
                        .alpha(0.38f),
                    contentScale = ContentScale.Crop
                )
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp)
        ) {
            // 상단 타이틀바
            Surface(
                color = Color(0xCC1A1A1A),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = uiState.characterName,
                            color = Color.White,
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold
                        )
                        if (uiState.currentRAGContext.isNotEmpty()) {
                            Text(
                                text = uiState.currentRAGContext,
                                color = Color(0xFFFFB6C1),
                                fontSize = 10.sp
                            )
                        }
                    }

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        IconButton(onClick = { filePickerLauncher.launch("application/json") }) {
                            Icon(Icons.Default.Add, contentDescription = "카드 가져오기", tint = Color.LightGray)
                        }
                        IconButton(onClick = onClearHistory) {
                            Icon(Icons.Default.Refresh, contentDescription = "대화 초기화", tint = Color.LightGray)
                        }
                        IconButton(onClick = { onToggleSettings(true) }) {
                            Icon(Icons.Default.Settings, contentDescription = "설정", tint = Color.LightGray)
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // 비주얼 노벨 본문 대화 리스트
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(uiState.messages, key = { it.id }) { msg ->
                    VisualNovelMessageItem(message = msg)
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // 하단 입력 창
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = { Text("대화를 입력하세요...", color = Color.Gray) },
                    modifier = Modifier.weight(1f),
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color(0xDD1E1E1E),
                        unfocusedContainerColor = Color(0xDD1E1E1E),
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White
                    ),
                    shape = RoundedCornerShape(8.dp)
                )

                Spacer(modifier = Modifier.width(8.dp))

                Button(
                    onClick = {
                        onSendMessage(inputText)
                        inputText = ""
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF8B0000)),
                    shape = RoundedCornerShape(8.dp),
                    enabled = !uiState.isStreaming
                ) {
                    if (uiState.isStreaming) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            color = Color.White,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text("전송", color = Color.White)
                    }
                }
            }
        }

        // API 키 설정 모달 다이얼로그
        if (uiState.showSettingsDialog) {
            SettingsDialog(
                currentOpenAiKey = uiState.openAiApiKey,
                currentGrokKey = uiState.grokApiKey,
                onDismiss = { onToggleSettings(false) },
                onSave = onSaveApiKeys
            )
        }
    }
}

/**
 * 대사 / 지문 / 속마음 정규식 기반 스타일링 파서
 */
@Composable
fun VisualNovelMessageItem(message: ChatUiMessage) {
    val isUser = message.sender == "user"

    Surface(
        color = if (isUser) Color(0xAA2A2A2A) else Color(0xEE121212),
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = if (isUser) "당신" else "캐릭터",
                color = if (isUser) Color(0xFFCCCCCC) else Color(0xFFFFB6C1),
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = parseVisualNovelText(message.text),
                fontSize = 14.sp,
                lineHeight = 21.sp
            )
        }
    }
}

/**
 * 한국어 및 서브컬처 기호 정규식 완벽 파싱 함수
 */
fun parseVisualNovelText(rawText: String): AnnotatedString {
    return buildAnnotatedString {
        var cursor = 0
        // 정규식 매칭: 대사("...", “...”), 행동/지문(「...」, 『...』, *...*), 속마음([...])
        val regex = Regex("(\"[^\"]*\"|“[^”]*”)|(「[^」]*」|『[^』]*』|\\*[^*]*\\*)|(\\[[^\\]]*\\])")
        val matches = regex.findAll(rawText)

        for (match in matches) {
            if (match.range.first > cursor) {
                append(rawText.substring(cursor, match.range.first))
            }

            val value = match.value
            when {
                // 1. 대사 ("..." / “...”): 고대비 Bold 폰트
                value.startsWith("\"") || value.startsWith("“") -> {
                    withStyle(SpanStyle(color = Color.White, fontWeight = FontWeight.Bold)) {
                        append(value)
                    }
                }
                // 2. 지문 (「...」 / 『...』 / *...*): 이탤릭 로즈핑크
                value.startsWith("「") || value.startsWith("『") || value.startsWith("*") -> {
                    withStyle(SpanStyle(color = Color(0xFFFFB6C1), fontStyle = FontStyle.Italic)) {
                        append(value)
                    }
                }
                // 3. 속마음 ([...]): 이탤릭 연회색
                value.startsWith("[") -> {
                    withStyle(SpanStyle(color = Color.LightGray, fontStyle = FontStyle.Italic)) {
                        append(value)
                    }
                }
                else -> append(value)
            }
            cursor = match.range.last + 1
        }

        if (cursor < rawText.length) {
            append(rawText.substring(cursor))
        }
    }
}

/**
 * API 키 설정 모달 다이얼로그 Composable
 */
@Composable
fun SettingsDialog(
    currentOpenAiKey: String,
    currentGrokKey: String,
    onDismiss: () -> Unit,
    onSave: (String, String) -> Unit
) {
    var openAiKeyInput by remember { mutableStateOf(currentOpenAiKey) }
    var grokKeyInput by remember { mutableStateOf(currentGrokKey) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("API 키 및 설정", color = Color.White) },
        text = {
            Column {
                Text("OpenAI API Key (로컬 RAG 임베딩용)", fontSize = 12.sp, color = Color.LightGray)
                OutlinedTextField(
                    value = openAiKeyInput,
                    onValueChange = { openAiKeyInput = it },
                    placeholder = { Text("sk-...") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text("xAI Grok API Key (대화 및 이미지 생성용)", fontSize = 12.sp, color = Color.LightGray)
                OutlinedTextField(
                    value = grokKeyInput,
                    onValueChange = { grokKeyInput = it },
                    placeholder = { Text("xai-...") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onSave(openAiKeyInput, grokKeyInput) },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF8B0000))
            ) {
                Text("저장")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("취소", color = Color.Gray)
            }
        },
        containerColor = Color(0xFF1E1E1E)
    )
}

