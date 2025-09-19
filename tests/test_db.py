from datetime import datetime


# Test user functions
def test_user_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Test insert_user
    user = test_db.insert_user(
        email="test@example.com",
        user_name="Test User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )
    assert user.id is not None
    assert user.email == "test@example.com"

    # Test get_user_by_id
    retrieved_user = test_db.get_user_by_id(user.id)
    assert retrieved_user is not None
    assert retrieved_user.id == user.id
    assert retrieved_user.email == "test@example.com"
    assert retrieved_user.user_name == "Test User"

    # Test get_user_by_email
    retrieved_user = test_db.get_user_by_email("test@example.com")
    assert retrieved_user is not None
    assert retrieved_user.id == user.id
    assert retrieved_user.email == "test@example.com"
    assert retrieved_user.user_name == "Test User"

    # Test update_user
    updated_user = test_db.update_user(user_id=user.id, user_name="Updated Name")
    assert updated_user.user_name == "Updated Name"

    # Verify update
    retrieved_user = test_db.get_user_by_id(user.id)
    assert retrieved_user.user_name == "Updated Name"

    test_db.insert_user(
        email="test2@example.com",
        user_name="Test User 2",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    # Test fetch_all_users
    all_users = test_db.fetch_all_users()
    assert len(all_users) == 2

    # Test get_user_by_id with non-existent user
    non_existent_user = test_db.get_user_by_id(9999)
    assert non_existent_user is None

    # Test get_user_by_email with non-existent email
    non_existent_user = test_db.get_user_by_email("nonexistent@example.com")
    assert non_existent_user is None


# Test course functions
def test_course_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a test user
    user = test_db.insert_user(
        email="course_test@example.com",
        user_name="Course Test User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    # Test insert_course
    course = test_db.insert_course(
        title="Test Course",
        description="A test course",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    assert course.id is not None
    assert course.title == "Test Course"

    # Test users_courses before enrollment
    enrolled, not_enrolled = test_db.users_courses(user.id)
    assert len(enrolled) == 0
    assert len(not_enrolled) >= 1
    assert any(c.id == course.id for c in not_enrolled)

    # Test enrollment
    enrollment = test_db.insert_enrollment(
        user_id=user.id,
        course_id=course.id,
        enrolled_at=now,
        completed_at=None,
        deleted_at=None,
    )
    assert enrollment.id is not None

    # Test is_enrolled
    assert test_db.is_enrolled(user.id, course.id) is True

    # Test users_courses after enrollment
    enrolled, not_enrolled = test_db.users_courses(user.id)
    assert len(enrolled) >= 1
    assert any(c.id == course.id for c in enrolled)

    # Test enrollment count
    enrollment_count = test_db.get_course_enrollment_count(course.id)
    assert enrollment_count >= 1

    # Test is_enrolled with non-enrolled user
    non_enrolled_user = test_db.insert_user(
        email="not_enrolled@example.com",
        user_name="Not Enrolled User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )
    assert test_db.is_enrolled(non_enrolled_user.id, course.id) is False

    # Test is_enrolled with non-existent course
    assert test_db.is_enrolled(user.id, 9999) is False


# Test section functions
def test_section_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a test course
    course = test_db.insert_course(
        title="Section Test Course",
        description="A test course for sections",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test insert_section
    section = test_db.insert_section(
        course_id=course.id,
        title="Test Section",
        description="A test section",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    assert section.id is not None
    assert section.title == "Test Section"

    # Test get_sections_by_course_id
    sections = test_db.get_sections_by_course_id(course.id)
    assert len(sections) >= 1
    assert any(s.id == section.id for s in sections)

    # Test get_sections_by_course_id for non-existent course
    empty_sections = test_db.get_sections_by_course_id(9999)
    assert len(empty_sections) == 0


# Add more test functions for other database operations
def test_lesson_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test course and section
    course = test_db.insert_course(
        title="Lesson Test Course",
        description="A test course for lessons",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    section = test_db.insert_section(
        course_id=course.id,
        title="Lesson Test Section",
        description="A test section for lessons",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test insert_lesson
    lesson = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Test Lesson",
        description="A test lesson",
        video_url="https://example.com/video",
        thumbnail="lesson_thumbnail.jpg",
        duration="10:00",
        content="Lesson content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    assert lesson.id is not None
    assert lesson.title == "Test Lesson"

    # Test get_lesson_by_id
    retrieved_lesson = test_db.get_lesson_by_id(lesson.id)
    assert retrieved_lesson is not None
    assert retrieved_lesson.id == lesson.id

    # Test get_lessons_by_course_id
    lessons = test_db.get_lessons_by_course_id(course.id)
    assert len(lessons) >= 1
    assert any(l.id == lesson.id for l in lessons)

    # Test get_lessons_by_section_id
    section_lessons = test_db.get_lessons_by_section_id(section.id)
    assert len(section_lessons) >= 1
    assert any(l.id == lesson.id for l in section_lessons)

    # Test get_lesson_by_id with non-existent lesson
    non_existent_lesson = test_db.get_lesson_by_id(9999)
    assert non_existent_lesson is None

    # Test get_lessons_by_course_id with non-existent course
    no_lessons = test_db.get_lessons_by_course_id(9999)
    assert len(no_lessons) == 0

    # Test get_lessons_by_section_id with non-existent section
    no_section_lessons = test_db.get_lessons_by_section_id(9999)
    assert len(no_section_lessons) == 0


def test_lesson_navigation(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test course
    course = test_db.insert_course(
        title="Navigation Test Course",
        description="A test course for lesson navigation",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    section = test_db.insert_section(
        course_id=course.id,
        title="Navigation Test Section",
        description="A test section for lesson navigation",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Create three lessons with sequential sort orders
    lesson1 = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Lesson 1",
        description="First lesson",
        video_url="https://example.com/video1",
        thumbnail="thumbnail1.jpg",
        duration="10:00",
        content="Lesson 1 content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson2 = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Lesson 2",
        description="Second lesson",
        video_url="https://example.com/video2",
        thumbnail="thumbnail2.jpg",
        duration="15:00",
        content="Lesson 2 content",
        sort_order=2,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson3 = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Lesson 3",
        description="Third lesson",
        video_url="https://example.com/video3",
        thumbnail="thumbnail3.jpg",
        duration="20:00",
        content="Lesson 3 content",
        sort_order=3,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test get_next_lesson from first lesson
    next_lesson = test_db.get_next_lesson(course.id, lesson1.sort_order)
    assert next_lesson is not None
    assert next_lesson.id == lesson2.id

    # Test get_next_lesson from middle lesson
    next_lesson = test_db.get_next_lesson(course.id, lesson2.sort_order)
    assert next_lesson is not None
    assert next_lesson.id == lesson3.id

    # Test get_next_lesson from last lesson (should return None)
    next_lesson = test_db.get_next_lesson(course.id, lesson3.sort_order)
    assert next_lesson is None

    # Test get_previous_lesson from last lesson
    prev_lesson = test_db.get_previous_lesson(course.id, lesson3.sort_order)
    assert prev_lesson is not None
    assert prev_lesson.id == lesson2.id

    # Test get_previous_lesson from middle lesson
    prev_lesson = test_db.get_previous_lesson(course.id, lesson2.sort_order)
    assert prev_lesson is not None
    assert prev_lesson.id == lesson1.id

    # Test get_previous_lesson from first lesson (should return None)
    prev_lesson = test_db.get_previous_lesson(course.id, lesson1.sort_order)
    assert prev_lesson is None

    # Test with non-existent course
    next_lesson = test_db.get_next_lesson(9999, 1)
    assert next_lesson is None
    prev_lesson = test_db.get_previous_lesson(9999, 1)
    assert prev_lesson is None


# Test conversation and message functions
def test_conversation_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test user, course, section, and lesson
    user = test_db.insert_user(
        email="conversation_test@example.com",
        user_name="Conversation Test User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    course = test_db.insert_course(
        title="Conversation Test Course",
        description="A test course",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    section = test_db.insert_section(
        course_id=course.id,
        title="Conversation Test Section",
        description="A test section",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Conversation Test Lesson",
        description="A test lesson",
        video_url="https://example.com/video",
        thumbnail="thumbnail.jpg",
        duration="10:00",
        content="Lesson content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test insert_conversation
    conversation = test_db.insert_conversation(
        user_id=user.id,
        lesson_id=lesson.id,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    assert conversation.id is not None

    # Test get_conversation_by_id_and_user_id
    retrieved_conversation = test_db.get_conversation_by_id_and_user_id(
        conversation.id, user.id
    )
    assert retrieved_conversation is not None
    assert retrieved_conversation.id == conversation.id

    # Test get_conversation_by_lesson_id_and_user_id
    retrieved_conversation = test_db.get_conversation_by_lesson_id_and_user_id(
        lesson.id, user.id
    )
    assert retrieved_conversation is not None
    assert retrieved_conversation.id == conversation.id

    # Test insert_message
    message = test_db.insert_message(
        conversation_id=conversation.id,
        user_id=user.id,
        content="Test message",
        role="user",
        created_at=now,
        deleted_at=None,
    )
    assert message.id is not None

    # Test conversation_messages
    messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(messages) >= 1
    assert any(m.id == message.id for m in messages)

    # Test update_message
    updated_message = test_db.update_message(
        message_id=message.id, content="Updated message"
    )
    assert updated_message.content == "Updated message"

    # Test update_conversation
    updated_conversation = test_db.update_conversation(
        conversation_id=conversation.id, updated_at=now
    )
    assert updated_conversation.updated_at == now

    # Test with non-existent conversation and user
    non_existent_conv = test_db.get_conversation_by_id_and_user_id(9999, user.id)
    assert non_existent_conv is None

    non_existent_conv = test_db.get_conversation_by_lesson_id_and_user_id(9999, user.id)
    assert non_existent_conv is None

    # Test conversation_messages with non-existent conversation
    empty_messages = test_db.conversation_messages(9999, user.id)
    assert len(empty_messages) == 0

    # Add messages with different roles
    user_message = test_db.insert_message(
        conversation_id=conversation.id,
        user_id=user.id,
        content="User message",
        role="user",
        created_at=now,
        deleted_at=None,
    )

    assistant_message = test_db.insert_message(
        conversation_id=conversation.id,
        user_id=user.id,
        content="Assistant message",
        role="assistant",
        created_at=now,
        deleted_at=None,
    )
    assert assistant_message.id is not None
    assert assistant_message.conversation_id == conversation.id
    assert assistant_message.user_id == user.id
    assert assistant_message.content == "Assistant message"
    assert assistant_message.role == "assistant"
    assert assistant_message.created_at == now
    assert assistant_message.deleted_at is None

    # Test messages retrieval has correct order
    messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(messages) >= 2
    # Check the last two messages match what we just inserted
    assert messages[-2].role == "user"
    assert messages[-2].content == "User message"
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Assistant message"

    # Test soft deletion of messages
    from db.db import messages as messages_table

    messages_table.update(id=user_message.id, deleted_at=now)

    # After deletion, should only see one message
    remaining_messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(remaining_messages) >= 1
    # Make sure the user message we deleted is not in the results
    assert not any(m.id == user_message.id for m in remaining_messages)
    # Make sure the assistant message is still there
    assert any(m.id == assistant_message.id for m in remaining_messages)

    # Test with different user shouldn't see messages
    other_user = test_db.insert_user(
        email="other_message_user@example.com",
        user_name="Other Message User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    other_user_messages = test_db.conversation_messages(conversation.id, other_user.id)
    assert len(other_user_messages) == 0


def test_lesson_completion_operations(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test user, course, section, and lesson
    user = test_db.insert_user(
        email="completion_test@example.com",
        user_name="Completion Test User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    course = test_db.insert_course(
        title="Completion Test Course",
        description="A test course for completions",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    section = test_db.insert_section(
        course_id=course.id,
        title="Completion Test Section",
        description="A test section for completions",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson1 = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Completion Test Lesson 1",
        description="A test lesson for completions",
        video_url="https://example.com/video1",
        thumbnail="thumbnail1.jpg",
        duration="10:00",
        content="Lesson 1 content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson2 = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Completion Test Lesson 2",
        description="A second test lesson for completions",
        video_url="https://example.com/video2",
        thumbnail="thumbnail2.jpg",
        duration="15:00",
        content="Lesson 2 content",
        sort_order=2,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test initial completion status (should be False)
    initial_status = test_db.get_lesson_completion_status(lesson1.id, user.id)
    assert initial_status is False

    # Test insert_lesson_completion with completed_at=None
    lesson_completion = test_db.insert_lesson_completion(
        user_id=user.id,
        lesson_id=lesson1.id,
        course_id=course.id,
        completed_at=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    assert lesson_completion.id is not None

    # Status should still be False since completed_at is None
    status_after_insert = test_db.get_lesson_completion_status(lesson1.id, user.id)
    assert status_after_insert is False

    # Test update_lesson_completion to set completed_at
    updated_completion = test_db.update_lesson_completion(
        lesson_completion_id=lesson_completion.id, completed_at=now
    )
    assert updated_completion.completed_at == now

    # Status should now be True
    status_after_update = test_db.get_lesson_completion_status(lesson1.id, user.id)
    assert status_after_update is True

    # Test get_lesson_completion_by_user_id_and_lesson_id
    retrieved_completion = test_db.get_lesson_completion_by_user_id_and_lesson_id(
        user.id, lesson1.id
    )
    assert retrieved_completion is not None
    assert retrieved_completion.id == lesson_completion.id

    # Test get_completed_lessons_for_user_id_and_course_id with one completion
    completed_count = test_db.get_completed_lessons_for_user_id_and_course_id(
        user.id, course.id
    )
    assert completed_count == 1

    # Complete another lesson
    lesson2_completion = test_db.insert_lesson_completion(
        user_id=user.id,
        lesson_id=lesson2.id,
        course_id=course.id,
        completed_at=now,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test count with two completions
    completed_count = test_db.get_completed_lessons_for_user_id_and_course_id(
        user.id, course.id
    )
    assert completed_count == 2

    # Test with non-existent lesson/user
    non_existent_status = test_db.get_lesson_completion_status(9999, user.id)
    assert non_existent_status is False

    non_existent_completion = test_db.get_lesson_completion_by_user_id_and_lesson_id(
        user.id, 9999
    )
    assert non_existent_completion is None

    # Test when lesson completion exists but was "uncompleted" (completed_at set back to None)
    test_db.update_lesson_completion(
        lesson_completion_id=lesson2_completion.id, completed_at=None
    )
    status_after_unmark = test_db.get_lesson_completion_status(lesson2.id, user.id)
    assert status_after_unmark is False

    # Count should now be back to 1
    completed_count = test_db.get_completed_lessons_for_user_id_and_course_id(
        user.id, course.id
    )
    assert completed_count == 1


def test_soft_deletion(test_db):
    # Setup
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    deleted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test entities
    user = test_db.insert_user(
        email="deletion_test@example.com",
        user_name="Deletion Test User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    course = test_db.insert_course(
        title="Deletion Test Course",
        description="Test course",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Insert enrollment
    enrollment = test_db.insert_enrollment(
        user_id=user.id,
        course_id=course.id,
        enrolled_at=now,
        completed_at=None,
        deleted_at=None,
    )

    # Verify enrollment exists
    assert test_db.is_enrolled(user.id, course.id) is True

    # Get enrollments table directly via the database module
    from db.db import enrollments

    # Soft delete user and enrollment
    test_db.update_user(user_id=user.id, deleted_at=deleted_time)
    enrollments.update(id=enrollment.id, deleted_at=deleted_time)

    # Verify soft-deleted enrollment is not found
    assert test_db.is_enrolled(user.id, course.id) is False

    # Test that course still appears in not_enrolled
    enrolled, not_enrolled = test_db.users_courses(user.id)
    assert len(enrolled) == 0
    assert any(c.id == course.id for c in not_enrolled)

    # Test section and lesson creation for a soft-deleted course
    # This shouldn't happen in practice but we should test DB behavior
    soft_deleted_course = test_db.insert_course(
        title="Soft Deleted Course",
        description="This course will be soft-deleted",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=deleted_time,  # Immediately marked as deleted
    )

    # Create section for soft-deleted course
    section = test_db.insert_section(
        course_id=soft_deleted_course.id,
        title="Section for deleted course",
        description="Test section",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Verify course doesn't appear in users_courses
    enrolled, not_enrolled = test_db.users_courses(user.id)
    assert not any(c.id == soft_deleted_course.id for c in enrolled)
    assert not any(c.id == soft_deleted_course.id for c in not_enrolled)

    # Sections for deleted course should still be retrievable directly
    sections = test_db.get_sections_by_course_id(soft_deleted_course.id)
    assert len(sections) >= 1
    assert any(s.id == section.id for s in sections)


def test_enrollment_edge_cases(test_db):
    """Test edge cases for enrollments and course status tracking."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test user and course
    user = test_db.insert_user(
        email="enrollment_edge@example.com",
        user_name="Enrollment Edge User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    course = test_db.insert_course(
        title="Enrollment Edge Course",
        description="A test course for enrollment edge cases",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test enrollment count for new course should be 0
    initial_count = test_db.get_course_enrollment_count(course.id)
    assert initial_count == 0

    # Test multiple enrollments for the same course
    enrollment1 = test_db.insert_enrollment(
        user_id=user.id,
        course_id=course.id,
        enrolled_at=now,
        completed_at=None,
        deleted_at=None,
    )

    # Create another user to enroll
    user2 = test_db.insert_user(
        email="enrollment_edge2@example.com",
        user_name="Enrollment Edge User 2",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    test_db.insert_enrollment(
        user_id=user2.id,
        course_id=course.id,
        enrolled_at=now,
        completed_at=None,
        deleted_at=None,
    )

    # Test count with two enrollments
    enrollment_count = test_db.get_course_enrollment_count(course.id)
    assert enrollment_count == 2

    # Test enrollment completion tracking
    from db.db import enrollments

    completed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enrollments.update(id=enrollment1.id, completed_at=completed_time)

    # Re-test users_courses after enrollment completion
    enrolled, not_enrolled = test_db.users_courses(user.id)
    assert len(enrolled) == 1
    assert any(c.id == course.id for c in enrolled)


def test_non_existent_course_section_relations(test_db):
    """Test handling of non-existent course/section relations"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test course
    course = test_db.insert_course(
        title="Non-existent Relations Course",
        description="Test handling of non-existent relations",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test with valid section
    section = test_db.insert_section(
        course_id=course.id,
        title="Test Section",
        description="Test section",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Create a standard lesson
    lesson = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Standard Lesson",
        description="A standard lesson",
        video_url="https://example.com/video",
        thumbnail="thumbnail.jpg",
        duration="10:00",
        content="Lesson content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Verify retrieval by course and section
    course_lessons = test_db.get_lessons_by_course_id(course.id)
    assert len(course_lessons) >= 1
    assert any(l.id == lesson.id for l in course_lessons)

    section_lessons = test_db.get_lessons_by_section_id(section.id)
    assert len(section_lessons) >= 1
    assert any(l.id == lesson.id for l in section_lessons)

    # Create another course to test multi-course scenarios
    other_course = test_db.insert_course(
        title="Other Course",
        description="Another test course",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Create section for other course
    other_section = test_db.insert_section(
        course_id=other_course.id,
        title="Other Section",
        description="Section for other course",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Create a lesson for other course
    test_db.insert_lesson(
        course_id=other_course.id,
        section_id=other_section.id,
        title="Other Course Lesson",
        description="Lesson for another course",
        video_url="https://example.com/video",
        thumbnail="thumbnail.jpg",
        duration="10:00",
        content="Lesson content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test getting non-existent lessons for a course
    non_existent_course_id = 9999
    non_existent_course_lessons = test_db.get_lessons_by_course_id(
        non_existent_course_id
    )
    assert len(non_existent_course_lessons) == 0

    # Test get_sections_by_course_id doesn't return sections from other courses
    course_sections = test_db.get_sections_by_course_id(course.id)
    assert any(s.id == section.id for s in course_sections)
    assert not any(s.id == other_section.id for s in course_sections)

    # Test lesson navigation between courses
    # Should not find next lesson from different course
    next_lesson = test_db.get_next_lesson(course.id, lesson.sort_order)
    assert next_lesson is None or next_lesson.course_id == course.id


def test_message_and_conversation_edge_cases(test_db):
    """Test edge cases for messages and conversations."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create test user, course, and lesson
    user = test_db.insert_user(
        email="message_edge@example.com",
        user_name="Message Edge User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    course = test_db.insert_course(
        title="Message Edge Course",
        description="A test course for message edge cases",
        thumbnail="thumbnail.jpg",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    section = test_db.insert_section(
        course_id=course.id,
        title="Message Edge Section",
        description="A test section",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    lesson = test_db.insert_lesson(
        course_id=course.id,
        section_id=section.id,
        title="Message Edge Lesson",
        description="A test lesson",
        video_url="https://example.com/video",
        thumbnail="thumbnail.jpg",
        duration="10:00",
        content="Lesson content",
        sort_order=1,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Create conversation without messages
    conversation = test_db.insert_conversation(
        user_id=user.id,
        lesson_id=lesson.id,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    # Test empty conversation_messages
    messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(messages) == 0

    # Add messages with different roles
    user_message = test_db.insert_message(
        conversation_id=conversation.id,
        user_id=user.id,
        content="User message",
        role="user",
        created_at=now,
        deleted_at=None,
    )

    assistant_message = test_db.insert_message(
        conversation_id=conversation.id,
        user_id=user.id,
        content="Assistant message",
        role="assistant",
        created_at=now,
        deleted_at=None,
    )
    assert assistant_message.id is not None
    assert assistant_message.conversation_id == conversation.id
    assert assistant_message.user_id == user.id
    assert assistant_message.content == "Assistant message"
    assert assistant_message.role == "assistant"
    assert assistant_message.created_at == now
    assert assistant_message.deleted_at is None

    # Test messages retrieval has correct order
    messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(messages) >= 2
    # Check the last two messages match what we just inserted
    assert messages[-2].role == "user"
    assert messages[-2].content == "User message"
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Assistant message"

    # Test soft deletion of messages
    from db.db import messages as messages_table

    messages_table.update(id=user_message.id, deleted_at=now)

    # After deletion, should only see one message
    remaining_messages = test_db.conversation_messages(conversation.id, user.id)
    assert len(remaining_messages) >= 1
    # Make sure the user message we deleted is not in the results
    assert not any(m.id == user_message.id for m in remaining_messages)
    # Make sure the assistant message is still there
    assert any(m.id == assistant_message.id for m in remaining_messages)

    # Test with different user shouldn't see messages
    other_user = test_db.insert_user(
        email="other_message_user@example.com",
        user_name="Other Message User",
        password="hashed_password",
        created_at=now,
        last_active=now,
    )

    other_user_messages = test_db.conversation_messages(conversation.id, other_user.id)
    assert len(other_user_messages) == 0
