CREATE TABLE `members` (
  `id` INT PRIMARY KEY AUTO_INCREMENT COMMENT '회원번호',
  `name` VARCHAR(50) NOT NULL COMMENT '회원이름'
);

CREATE TABLE `books` (
  `id` INT PRIMARY KEY AUTO_INCREMENT COMMENT '도서번호',
  `title` VARCHAR(100) NOT NULL COMMENT '도서제목'
);

CREATE TABLE `loans` (
  `id` INT PRIMARY KEY AUTO_INCREMENT COMMENT '대출번호',
  `member_id` INT NOT NULL COMMENT '대출한 회원',
  `book_id` INT NOT NULL COMMENT '대출된 도서',
  `loan_date` DATETIME NOT NULL COMMENT '대출일시',
  `due_date` DATE NOT NULL COMMENT '반납 예정일',
  `return_date` DATE COMMENT '실제 반납일',
  `status` ENUM ('A', 'B', 'C') NOT NULL DEFAULT "A" COMMENT '대출상태(A=대출중,B=반납완료,C=연체)',
  `fine_amount` decimal(10,2) DEFAULT 0 COMMENT '연체료',
  `auto_renewal` boolean DEFAULT false COMMENT '반납일 자동 연장 여부',
  `notes` text COMMENT '메모(특이사항)'
);

ALTER TABLE `loans` ADD FOREIGN KEY (`member_id`) REFERENCES `members` (`id`);

ALTER TABLE `loans` ADD FOREIGN KEY (`book_id`) REFERENCES `books` (`id`);

ALTER TABLE `members` COMMENT = '회원';

ALTER TABLE `books` COMMENT = '도서';

ALTER TABLE `loans` COMMENT = '도서 대출';
