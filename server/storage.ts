export type User = {
  id: number;
  username: string;
  password: string;
};

export type InsertUser = Omit<User, "id">;

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
}

export class MemoryStorage implements IStorage {
  private users = new Map<number, User>();
  private nextId = 1;

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find((user) => user.username === username);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const user = { ...insertUser, id: this.nextId++ };
    this.users.set(user.id, user);
    return user;
  }
}

export const storage = new MemoryStorage();
