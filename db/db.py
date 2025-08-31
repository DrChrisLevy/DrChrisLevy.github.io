# ruff: noqa: F403, F405

import os

from fasthtml.common import database

# Database setup
DB_PATH = os.environ.get("DB_PATH", "data/teaching.db")
db = database(DB_PATH)
users = db.t.users
messages = db.t.messages
conversations = db.t.conversations
courses = db.t.courses
lessons = db.t.lessons
enrollments = db.t.enrollments
sections = db.t.sections
lesson_completions = db.t.lesson_completions
# create courses table if it doesn't exist
if courses not in db.t:
    courses.create(
        id=int,  # primary key
        title=str,  # course title
        description=str,  # course description
        thumbnail=str,  # thumbnail image URL
        created_at=str,  # creation timestamp
        updated_at=str,  # last update timestamp
        deleted_at=str,  # deletion timestamp (for soft delete)
        pk="id",
    )

if sections not in db.t:
    sections.create(
        id=int,  # primary key
        course_id=int,  # foreign key to courses.id
        title=str,  # section title
        description=str,  # section description
        sort_order=int,  # section sort_order
        created_at=str,  # creation timestamp
        updated_at=str,  # last update timestamp
        deleted_at=str,  # deletion timestamp (for soft delete)
        pk="id",
        foreign_keys=[("course_id", "courses", "id")],
    )

# create lessons table if it doesn't exist
if lessons not in db.t:
    lessons.create(
        id=int,  # primary key
        course_id=int,  # foreign key to courses.id
        title=str,  # lesson title
        description=str,  # lesson description
        video_url=str,  # video URL
        thumbnail=str,  # thumbnail image URL
        duration=str,  # duration of the lesson
        created_at=str,  # creation timestamp
        updated_at=str,  # last update timestamp
        deleted_at=str,  # deletion timestamp (for soft delete)
        content=str,  # lesson content
        system_prompt=str,  # system prompt for the LLM chat for this lesson
        section_id=int,  # foreign key to section.id
        sort_order=int,  # lesson sort_order
        pk="id",
        foreign_keys=[("course_id", "courses", "id"), ("section_id", "sections", "id")],
    )


# Create users table if it doesn't exist
if users not in db.t:
    users.create(
        id=int,  # primary key
        email=str,  # login identifier
        user_name=str,  # display name
        password=str,  # hashed password
        created_at=str,  # account creation timestamp
        last_active=str,  # last request logic timestamp
        deleted_at=str,  # deletion timestamp (for soft delete)
        pk="id",
    )

if conversations not in db.t:
    conversations.create(
        id=int,  # primary key
        user_id=int,  # foreign key to users.id
        lesson_id=int,  # foreign key to lessons.id
        created_at=str,  # timestamp
        updated_at=str,  # timestamp
        deleted_at=str,  # timestamp
        pk="id",
        foreign_keys=[
            ("user_id", "users", "id"),
            ("lesson_id", "lessons", "id"),
        ],
    )
# Create messages table if it doesn't exist
if messages not in db.t:
    messages.create(
        id=int,  # primary key
        conversation_id=int,  # foreign key to conversations.id
        user_id=int,  # foreign key to users.id
        content=str,  # actual message content
        role=str,  # 'user' or 'assistant'
        created_at=str,  # timestamp
        deleted_at=str,  # timestamp
        pk="id",
        foreign_keys=[
            ("conversation_id", "conversations", "id"),
            ("user_id", "users", "id"),
        ],
    )

# create enrollments table if it doesn't exist
if enrollments not in db.t:
    enrollments.create(
        id=int,  # primary key
        user_id=int,  # foreign key to users.id
        course_id=int,  # foreign key to courses.id
        enrolled_at=str,  # enrollment timestamp
        completed_at=str,  # completion timestamp (null if not completed)
        deleted_at=str,  # deletion timestamp (for soft delete)
        pk="id",
        foreign_keys=[
            ("user_id", "users", "id"),
            ("course_id", "courses", "id"),
        ],
    )

# create lesson_completions table if it doesn't exist
if lesson_completions not in db.t:
    lesson_completions.create(
        id=int,  # primary key
        user_id=int,  # foreign key to users.id
        lesson_id=int,  # foreign key to lessons.id
        course_id=int,  # foreign key to courses.id
        completed_at=str,  # completion timestamp (null if not completed)
        created_at=str,  # creation timestamp
        updated_at=str,  # last update timestamp
        deleted_at=str,  # deletion timestamp (for soft delete)
        pk="id",
        foreign_keys=[
            ("user_id", "users", "id"),
            ("lesson_id", "lessons", "id"),
            ("course_id", "courses", "id"),
        ],
    )

# Create dataclass types for the tables
(
    User,
    Message,
    Conversation,
    Course,
    Lesson,
    Enrollment,
    LessonCompletion,
    CourseSection,
) = (
    users.dataclass(),
    messages.dataclass(),
    conversations.dataclass(),
    courses.dataclass(),
    lessons.dataclass(),
    enrollments.dataclass(),
    lesson_completions.dataclass(),
    sections.dataclass(),
)


class DataBase:
    # -- Users --
    @classmethod
    def get_user_by_id(cls, user_id: int) -> User | None:
        user = next(users.rows_where("id = ?", [user_id]), None)
        user = User(**user) if user else None
        return user

    @classmethod
    def get_user_by_email(cls, email: str) -> User | None:
        user = next(users.rows_where("email = ?", [email]), None)
        user = User(**user) if user else None
        return user

    @classmethod
    def update_user(cls, user_id: int, **kwargs) -> User:
        u = users.update(id=user_id, **kwargs)
        return u

    @classmethod
    def insert_user(cls, **kwargs) -> User:
        return users.insert(**kwargs)

    @classmethod
    def fetch_all_users(cls) -> list[User]:
        return users()

    # -- Courses --
    @classmethod
    def users_courses(cls, user_id: int) -> tuple[list[Course], list[Course]]:
        courses = db.query(
            """
            SELECT c.*,
                CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END as is_enrolled
            FROM courses c
            LEFT JOIN enrollments e ON c.id = e.course_id AND e.user_id = ? AND e.deleted_at IS NULL
            WHERE c.deleted_at IS NULL
            """,
            [user_id],
        )
        enrolled_courses = []
        not_enrolled_courses = []

        for course in courses:
            if course.pop("is_enrolled"):
                enrolled_courses.append(course)
            else:
                not_enrolled_courses.append(course)
        enrolled_courses = [Course(**c) for c in enrolled_courses]
        not_enrolled_courses = [Course(**c) for c in not_enrolled_courses]
        return enrolled_courses, not_enrolled_courses

    @classmethod
    def insert_course(cls, **kwargs) -> Course:
        return courses.insert(**kwargs)

    # -- Messages and Conversations --
    @classmethod
    def conversation_messages(cls, conversation_id: int, user_id: int) -> list[Message]:
        return [
            Message(**m)
            for m in messages.rows_where(
                "conversation_id = ? AND user_id = ? AND deleted_at IS NULL ORDER BY created_at ASC",
                [conversation_id, user_id],
            )
        ]

    @classmethod
    def update_message(cls, message_id: int, **kwargs) -> Message:
        return messages.update(id=message_id, **kwargs)

    @classmethod
    def update_conversation(cls, conversation_id: int, **kwargs) -> Conversation:
        return conversations.update(id=conversation_id, **kwargs)

    @classmethod
    def insert_message(cls, **kwargs) -> Message:
        return messages.insert(**kwargs)

    @classmethod
    def insert_conversation(cls, **kwargs) -> Conversation:
        return conversations.insert(**kwargs)

    @classmethod
    def get_conversation_by_id_and_user_id(
        cls, conversation_id: int, user_id: int
    ) -> Conversation | None:
        c = next(
            conversations.rows_where(
                "id = ? AND user_id = ? AND deleted_at IS NULL",
                [conversation_id, user_id],
            ),
            None,
        )
        return Conversation(**c) if c else None

    @classmethod
    def get_conversation_by_lesson_id_and_user_id(
        cls, lesson_id: int, user_id: int
    ) -> Conversation | None:
        c = next(
            conversations.rows_where(
                "lesson_id = ? AND user_id = ? AND deleted_at IS NULL",
                [lesson_id, user_id],
            ),
            None,
        )
        return Conversation(**c) if c else None

    # -- Lessons and Sections --

    @classmethod
    def insert_lesson(cls, **kwargs) -> Lesson:
        return lessons.insert(**kwargs)

    @classmethod
    def insert_section(cls, **kwargs) -> CourseSection:
        return sections.insert(**kwargs)

    @classmethod
    def get_lesson_by_id(cls, lesson_id: int) -> Lesson | None:
        l = next(lessons.rows_where("id = ?", [lesson_id]), None)
        return Lesson(**l) if l else None

    @classmethod
    def get_lessons_by_course_id(cls, course_id: int) -> list[Lesson]:
        return [
            Lesson(**l)
            for l in lessons.rows_where(
                "course_id = ? AND deleted_at IS NULL order by sort_order", [course_id]
            )
        ]

    @classmethod
    def get_lessons_by_section_id(cls, section_id: int) -> list[Lesson]:
        return [
            Lesson(**l)
            for l in lessons.rows_where(
                "section_id = ? AND deleted_at IS NULL order by sort_order",
                [section_id],
            )
        ]

    @classmethod
    def get_next_lesson(
        cls, course_id: int, current_lesson_sort_order: int
    ) -> Lesson | None:
        next_lesson = next(
            lessons.rows_where(
                "course_id = ? and sort_order > ? order by sort_order",
                [course_id, current_lesson_sort_order],
            ),
            None,
        )
        return Lesson(**next_lesson) if next_lesson else None

    @classmethod
    def get_previous_lesson(
        cls, course_id: int, current_lesson_sort_order: int
    ) -> Lesson | None:
        previous_lesson = next(
            lessons.rows_where(
                "course_id = ? and sort_order < ? order by sort_order desc",
                [course_id, current_lesson_sort_order],
            ),
            None,
        )
        return Lesson(**previous_lesson) if previous_lesson else None

    @classmethod
    def get_sections_by_course_id(cls, course_id: int) -> list[CourseSection]:
        return [
            CourseSection(**s)
            for s in sections.rows_where(
                "course_id = ? AND deleted_at IS NULL order by sort_order", [course_id]
            )
        ]

    # --- Lesson Completions ---
    @classmethod
    def insert_lesson_completion(cls, **kwargs) -> LessonCompletion:
        return lesson_completions.insert(**kwargs)

    @classmethod
    def update_lesson_completion(
        cls, lesson_completion_id: int, **kwargs
    ) -> LessonCompletion:
        return lesson_completions.update(id=lesson_completion_id, **kwargs)

    @classmethod
    def get_lesson_completion_status(cls, lesson_id: int, user_id: int) -> bool:
        row = next(
            lesson_completions.rows_where(
                "lesson_id = ? AND user_id = ?", [lesson_id, user_id]
            ),
            None,
        )
        if not row:
            return False

        if row["completed_at"] is None:
            return False
        else:
            return True

    @classmethod
    def get_lesson_completion_by_user_id_and_lesson_id(
        cls, user_id: int, lesson_id: int
    ) -> LessonCompletion | None:
        lc = next(
            lesson_completions.rows_where(
                "lesson_id = ? AND user_id = ?", [lesson_id, user_id]
            ),
            None,
        )
        return LessonCompletion(**lc) if lc else None

    @classmethod
    def get_completed_lessons_for_user_id_and_course_id(
        cls, user_id: int, course_id: int
    ) -> int:
        return len(
            list(
                lesson_completions.rows_where(
                    "course_id = ? AND user_id = ? AND deleted_at IS NULL and completed_at is not null",
                    [course_id, user_id],
                )
            )
        )

    # -- Enrollments--
    @classmethod
    def insert_enrollment(cls, **kwargs) -> Enrollment:
        return enrollments.insert(**kwargs)

    @classmethod
    def is_enrolled(cls, user_id: int, course_id: int) -> bool:
        rec = next(
            enrollments.rows_where(
                "course_id = ? AND user_id = ? AND deleted_at IS NULL",
                [course_id, user_id],
            ),
            None,
        )
        return rec is not None

    @classmethod
    def get_course_enrollment_count(cls, course_id: int) -> int:
        return len(
            list(
                enrollments.rows_where(
                    "course_id = ? AND deleted_at IS NULL", [course_id]
                )
            )
        )
