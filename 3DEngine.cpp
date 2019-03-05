#include "olcConsoleGameEngine.h"
#include <iostream>
#include <fstream>
#include <strstream>
#include <algorithm>
using namespace std;

struct vec3d
{
public:
	//Methods:
	float magnitude() const 
	{
		float mag = sqrtf(this->x*this->x + this->y*this->y + this->z*this->z);
		return mag;
	}

	vec3d normalize() const
	{
		float mag = magnitude();
		return { this->x / mag, this->y / mag, this->z / mag };
	}

	vec3d cross(const vec3d& other) const
	{
		vec3d normal;
		normal.x = this->y * other.z - this->z * other.y;
		normal.y = this->z * other.x - this->x * other.z;
		normal.z = this->x * other.y - this->y * other.x;
		return normal;
	}

	float dot(const vec3d& other) const
	{
		float result = this->x*other.x + this->y*other.y + this->z*other.z;
		return result;
	}

	//Operators:
	vec3d operator+(const vec3d& other) const
	{
		vec3d result;
		result.x = this->x + other.x; result.y = this->y + other.y; result.z = this->z + other.z;
		return result;
	}

	vec3d operator-(const vec3d& other) const
	{
		vec3d result;
		result.x = this->x - other.x; result.y = this->y - other.y; result.z = this->z - other.z;
		return result;
	}

	vec3d operator*(float k) const 
	{
		vec3d result;
		result.x = this->x*k; result.y = this->y*k; result.z = this->z*k;
		return result;
	}

	vec3d operator/(float k) const 
	{
		vec3d result;
		result.x = this->x / k; result.y = this->y / k; result.z = this->z / k;
		return result;
	}

	//Attributes:
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 1.0f;
};

vec3d operator*(float k, const vec3d& vec) 
{
	return vec * k;
}

struct triangle //3 vertices of vector 3d's
{
public:
	//Methods:
	// Calculate normal vector of each triangle
	vec3d normal() const {
		vec3d normal, line1, line2;
		//Create two lines from two adjacent sides of the triangle
		line1 = this->p[1] - this->p[0];
		line2 = this->p[2] - this->p[0];

		//Dot product between the lines gives a vector that is normal to both
		normal = line1.cross(line2);

		//Scale to unit vector
		normal = normal.normalize();

		return normal;
	}

	//Attributes:
	vec3d p[3];
	wchar_t sym;
	short col;
};

struct mesh
{
	vector<triangle> tris;

	bool LoadFromObjectFile(string sFilename)
	{
		ifstream f(sFilename);
		if (!f.is_open())
			return false;

		// Local cache of verts
		vector<vec3d> verts;

		while (!f.eof())
		{
			char line[128];
			f.getline(line, 128);

			strstream s;
			s << line;

			char junk;

			if (line[0] == 'v')
			{
				vec3d v;
				s >> junk >> v.x >> v.y >> v.z;
				verts.push_back(v);
			}

			if (line[0] == 'f')
			{
				int f[3];
				s >> junk >> f[0] >> f[1] >> f[2];
				tris.push_back({ verts[f[0] - 1], verts[f[1] - 1], verts[f[2] - 1] });
			}
		}
		return true;
	}
};

struct mat4x4 
{
public:
	//Methods:
	vec3d multiplyVector(vec3d &i) const
	{
		vec3d v;
		v.x = i.x * this->m[0][0] + i.y * this->m[1][0] + i.z * this->m[2][0] + i.w * this->m[3][0];
		v.y = i.x * this->m[0][1] + i.y * this->m[1][1] + i.z * this->m[2][1] + i.w * this->m[3][1];
		v.z = i.x * this->m[0][2] + i.y * this->m[1][2] + i.z * this->m[2][2] + i.w * this->m[3][2];
		v.w = i.x * this->m[0][3] + i.y * this->m[1][3] + i.z * this->m[2][3] + i.w * this->m[3][3];
		return v;
	}
	
	triangle multiplyTriangle(triangle &i) const
	{
		triangle o;
		o.p[0] = multiplyVector(i.p[0]);
		o.p[1] = multiplyVector(i.p[1]);
		o.p[2] = multiplyVector(i.p[2]);
		return o;
	}

	mat4x4 multiplyMatrix(mat4x4& m2)
	{
		mat4x4 matrix;
		for (int c = 0; c < 4; c++)
			for (int r = 0; r < 4; r++)
				matrix.m[r][c] = this->m[r][0] * m2.m[0][c] + this->m[r][1] * m2.m[1][c] + this->m[r][2] * m2.m[2][c] + this->m[r][3] * m2.m[3][c];
		return matrix;
	}
	
	//Attributes:
	float m[4][4] = { 0 }; 
};

class olcEngine3D : public olcConsoleGameEngine
{
public:
	olcEngine3D() 
	{
		m_sAppName = L"3D Demo";
	}

	bool OnUserCreate() override 
	{
		
		meshCube.LoadFromObjectFile("axis.obj");

		//Projection Matrix
		matProj = matrix_MakeProjection(90.0f, (float)ScreenHeight() / (float)ScreenWidth(), 0.1f, 1000.0f);

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override 
	{
		//Player input
		if (GetKey(VK_UP).bHeld)
			vCamera.y += 8.0f * fElapsedTime;

		if (GetKey(VK_DOWN).bHeld)
			vCamera.y -= 8.0f * fElapsedTime;

		if (GetKey(VK_LEFT).bHeld)
			vCamera.x -= 8.0f * fElapsedTime;

		if (GetKey(VK_RIGHT).bHeld)
			vCamera.x += 8.0f * fElapsedTime;

		//Look direction, scaled by the speed we want to move at
		vec3d vForward = vLookDir * 8.0f*fElapsedTime;

		if (GetKey(L'W').bHeld)
			vCamera = vCamera + vForward;

		if (GetKey(L'S').bHeld)
			vCamera = vCamera - vForward;

		if (GetKey(L'A').bHeld)
			fYaw -= 2.0f * fElapsedTime;

		if (GetKey(L'D').bHeld)
			fYaw += 2.0f * fElapsedTime;


		//Clear Screen
		Fill(0, 0, ScreenWidth(), ScreenHeight(), PIXEL_SOLID, FG_BLACK);
		
		// Rotation matrices
		mat4x4 matRotZ, matRotX;
		//fTheta += 1.0f * fElapsedTime;

		matRotZ = matrix_MakeRotationZ(fTheta * 0.5f);
		matRotX = matrix_MakeRotationX(fTheta);

		// Translation matrix
		mat4x4 matTrans;
		matTrans = matrix_MakeTranslation(0.0f, 0.0f, 5.0f);

		// World matrix
		mat4x4 matWorld;
		matWorld = matrix_MakeIdentity();
		matWorld = matRotZ.multiplyMatrix(matRotX);   //Rotating the object    (second)
		matWorld = matWorld.multiplyMatrix(matTrans); //Translating the object (first)

		//Camera matrix
		vec3d vUp = { 0.0f, 0.1f, 0.0f };
		vec3d vTarget = { 0.0f, 0.0f, 1.0f };
		mat4x4 matCameraRot = matrix_MakeRotationY(fYaw);
		vLookDir = matCameraRot.multiplyVector(vTarget);
		vTarget = vCamera + vLookDir;

		mat4x4 matCamera = matrix_PointAt(vCamera, vTarget, vUp);
		mat4x4 matView = matrix_QuickInverse(matCamera);

		//Store triangles for rastering later
		vector<triangle> vecTrianglesToRaster;

		// Draw Triangles
		for(auto tri : meshCube.tris) 
		{
			triangle triProjected, triTransformed, triViewed;

			triTransformed = matWorld.multiplyTriangle(tri);

			// Calculate surface normal of each triangle
			vec3d normal = triTransformed.normal();
			normal = normal.normalize();

			//Cast a ray from camera to triangle
			vec3d vCameraRay = triTransformed.p[0] - vCamera;

			if (normal.dot(vCameraRay) < 0.0f) //Check if triangle is visible
			{
				//Illumination
				vec3d light_direction = { 0.0f, 1.0f, -1.0f };
				light_direction = light_direction.normalize();

				//How "aligned" are the light direction and the triangle's surface normal?
				float dp = max(0.1f, light_direction.dot(normal));

				//Choose console colors are required
				CHAR_INFO c = GetColour(dp);
				triTransformed.col = c.Attributes;
				triTransformed.sym = c.Char.UnicodeChar;

				//Convert World Space --> View (camera's) space
				triViewed = matView.multiplyTriangle(triTransformed);

				//Project 3D --> 2D
				triProjected = matProj.multiplyTriangle(triViewed); //Project coordinates
				triProjected.col = triTransformed.col; //Color info stays the same
				triProjected.sym = triTransformed.sym;

				//Normalize into cartesian space
				triProjected.p[0] = triProjected.p[0] / triProjected.p[0].w;
				triProjected.p[1] = triProjected.p[1] / triProjected.p[1].w;
				triProjected.p[2] = triProjected.p[2] / triProjected.p[2].w;

				// X/Y are inverted so put them back
				triProjected.p[0].x *= -1.0f;
				triProjected.p[1].x *= -1.0f;
				triProjected.p[2].x *= -1.0f;
				triProjected.p[0].y *= -1.0f;
				triProjected.p[1].y *= -1.0f;
				triProjected.p[2].y *= -1.0f;

				// Scale into view (offset each vertex of the triangle by the same offset vector)
				vec3d vOffsetView = { 1.0f, 1.0f, 0.0f };
				triProjected.p[0] = triProjected.p[0] + vOffsetView;
				triProjected.p[1] = triProjected.p[1] + vOffsetView;
				triProjected.p[2] = triProjected.p[2] + vOffsetView;

				triProjected.p[0].x *= 0.5f * (float)ScreenWidth();
				triProjected.p[0].y *= 0.5f * (float)ScreenHeight();
				triProjected.p[1].x *= 0.5f * (float)ScreenWidth();
				triProjected.p[1].y *= 0.5f * (float)ScreenHeight();
				triProjected.p[2].x *= 0.5f * (float)ScreenWidth();
				triProjected.p[2].y *= 0.5f * (float)ScreenHeight();

				// Store triangles for sorting
				vecTrianglesToRaster.push_back(triProjected);
			}
		}

		//Sort triangles from back to front (Painter's Algorithm)
		sort(vecTrianglesToRaster.begin(), vecTrianglesToRaster.end(), [](triangle &t1, triangle &t2) 
		{
			//If midpoint of t1's z-values is further back, true (i.e. sort from back to front).
			float z1 = (t1.p[0].z + t1.p[1].z + t1.p[2].z) / 3.0f;
			float z2 = (t2.p[0].z + t2.p[1].z + t2.p[2].z) / 3.0f;
			return z1 > z2;
		});

		for (auto &triProjected : vecTrianglesToRaster) 
		{
			//Rasterize triangle
			FillTriangle(triProjected.p[0].x, triProjected.p[0].y,
				triProjected.p[1].x, triProjected.p[1].y,
				triProjected.p[2].x, triProjected.p[2].y,
				triProjected.sym, triProjected.col);
			/*
			DrawTriangle(triProjected.p[0].x, triProjected.p[0].y,
				triProjected.p[1].x, triProjected.p[1].y,
				triProjected.p[2].x, triProjected.p[2].y,
				PIXEL_SOLID, FG_BLACK); */
		}

		return true;
	}

private:

	CHAR_INFO GetColour(float lum)
	{
		short bg_col, fg_col;
		wchar_t sym;
		int pixel_bw = (int)(13.0f*lum);
		switch (pixel_bw)
		{
		case 0: bg_col = BG_BLACK; fg_col = FG_BLACK; sym = PIXEL_SOLID; break;

		case 1: bg_col = BG_BLACK; fg_col = FG_DARK_GREY; sym = PIXEL_QUARTER; break;
		case 2: bg_col = BG_BLACK; fg_col = FG_DARK_GREY; sym = PIXEL_HALF; break;
		case 3: bg_col = BG_BLACK; fg_col = FG_DARK_GREY; sym = PIXEL_THREEQUARTERS; break;
		case 4: bg_col = BG_BLACK; fg_col = FG_DARK_GREY; sym = PIXEL_SOLID; break;

		case 5: bg_col = BG_DARK_GREY; fg_col = FG_GREY; sym = PIXEL_QUARTER; break;
		case 6: bg_col = BG_DARK_GREY; fg_col = FG_GREY; sym = PIXEL_HALF; break;
		case 7: bg_col = BG_DARK_GREY; fg_col = FG_GREY; sym = PIXEL_THREEQUARTERS; break;
		case 8: bg_col = BG_DARK_GREY; fg_col = FG_GREY; sym = PIXEL_SOLID; break;

		case 9:  bg_col = BG_GREY; fg_col = FG_WHITE; sym = PIXEL_QUARTER; break;
		case 10: bg_col = BG_GREY; fg_col = FG_WHITE; sym = PIXEL_HALF; break;
		case 11: bg_col = BG_GREY; fg_col = FG_WHITE; sym = PIXEL_THREEQUARTERS; break;
		case 12: bg_col = BG_GREY; fg_col = FG_WHITE; sym = PIXEL_SOLID; break;
		default:
			bg_col = BG_BLACK; fg_col = FG_BLACK; sym = PIXEL_SOLID;
		}

		CHAR_INFO c;
		c.Attributes = bg_col | fg_col;
		c.Char.UnicodeChar = sym;
		return c;
	}

	mat4x4 matrix_MakeIdentity() 
	{
		mat4x4 matrix;
		matrix.m[0][0] = 1.0f;
		matrix.m[1][1] = 1.0f;
		matrix.m[2][2] = 1.0f;
		matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mat4x4 matrix_MakeRotationX(float fAngleRad)
	{
		mat4x4 matrix;
		matrix.m[0][0] = 1.0f;
		matrix.m[1][1] = cosf(fAngleRad);
		matrix.m[1][2] = sinf(fAngleRad);
		matrix.m[2][1] = -sinf(fAngleRad);
		matrix.m[2][2] = cosf(fAngleRad);
		matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mat4x4 matrix_MakeRotationY(float fAngleRad)
	{
		mat4x4 matrix;
		matrix.m[0][0] = cosf(fAngleRad);
		matrix.m[0][2] = sinf(fAngleRad);
		matrix.m[2][0] = -sinf(fAngleRad);
		matrix.m[1][1] = 1.0f;
		matrix.m[2][2] = cosf(fAngleRad);
		matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mat4x4 matrix_MakeRotationZ(float fAngleRad)
	{
		mat4x4 matrix;
		matrix.m[0][0] = cosf(fAngleRad);
		matrix.m[0][1] = sinf(fAngleRad);
		matrix.m[1][0] = -sinf(fAngleRad);
		matrix.m[1][1] = cosf(fAngleRad);
		matrix.m[2][2] = 1.0f;
		matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mat4x4 matrix_MakeTranslation(float x, float y, float z)
	{
		mat4x4 matrix;
		matrix.m[0][0] = 1.0f;
		matrix.m[1][1] = 1.0f;
		matrix.m[2][2] = 1.0f;
		matrix.m[3][3] = 1.0f;
		matrix.m[3][0] = x;
		matrix.m[3][1] = y;
		matrix.m[3][2] = z;
		return matrix;
	}

	mat4x4 matrix_MakeProjection(float fFovDegrees, float fAspectRatio, float fNear, float fFar)
	{
		float fFovRad = 1.0f / tanf(fFovDegrees * 0.5f / 180.0f * 3.14159f);
		mat4x4 matrix;
		matrix.m[0][0] = fAspectRatio * fFovRad;
		matrix.m[1][1] = fFovRad;
		matrix.m[2][2] = fFar / (fFar - fNear);
		matrix.m[3][2] = (-fFar * fNear) / (fFar - fNear);
		matrix.m[2][3] = 1.0f;
		matrix.m[3][3] = 0.0f;
		return matrix;
	}

	//This matrix takes an initial point at "pos", 
	//and rotates/translates it to 
	mat4x4 matrix_PointAt(vec3d& pos, vec3d& target, vec3d& up) 
	{
		//Calculate new forward direction
		vec3d newForward = target - pos;
		newForward = newForward.normalize();

		//Calculate new up direction
		vec3d a = newForward * up.dot(newForward);
		vec3d newUp = up - a;
		newUp = newUp.normalize();

		//Calculate new right direction
		vec3d newRight = newUp.cross(newForward);

		// Construct Dimensioning and Translation Matrix	
		mat4x4 matrix;
		matrix.m[0][0] = newRight.x;	matrix.m[0][1] = newRight.y;	matrix.m[0][2] = newRight.z;	matrix.m[0][3] = 0.0f;
		matrix.m[1][0] = newUp.x;		matrix.m[1][1] = newUp.y;		matrix.m[1][2] = newUp.z;		matrix.m[1][3] = 0.0f;
		matrix.m[2][0] = newForward.x;	matrix.m[2][1] = newForward.y;	matrix.m[2][2] = newForward.z;	matrix.m[2][3] = 0.0f;
		matrix.m[3][0] = pos.x;			matrix.m[3][1] = pos.y;			matrix.m[3][2] = pos.z;			matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mat4x4 matrix_QuickInverse(mat4x4 &m) // Only for Rotation/Translation Matrices
	{
		mat4x4 matrix;
		matrix.m[0][0] = m.m[0][0]; matrix.m[0][1] = m.m[1][0]; matrix.m[0][2] = m.m[2][0]; matrix.m[0][3] = 0.0f;
		matrix.m[1][0] = m.m[0][1]; matrix.m[1][1] = m.m[1][1]; matrix.m[1][2] = m.m[2][1]; matrix.m[1][3] = 0.0f;
		matrix.m[2][0] = m.m[0][2]; matrix.m[2][1] = m.m[1][2]; matrix.m[2][2] = m.m[2][2]; matrix.m[2][3] = 0.0f;
		matrix.m[3][0] = -(m.m[3][0] * matrix.m[0][0] + m.m[3][1] * matrix.m[1][0] + m.m[3][2] * matrix.m[2][0]);
		matrix.m[3][1] = -(m.m[3][0] * matrix.m[0][1] + m.m[3][1] * matrix.m[1][1] + m.m[3][2] * matrix.m[2][1]);
		matrix.m[3][2] = -(m.m[3][0] * matrix.m[0][2] + m.m[3][1] * matrix.m[1][2] + m.m[3][2] * matrix.m[2][2]);
		matrix.m[3][3] = 1.0f;
		return matrix;
	}

	mesh meshCube;
	mat4x4 matProj;
	vec3d vCamera = { 0.0f, 0.0f, 0.0f };
	vec3d vLookDir;
	float fYaw;
	float fTheta = 0.0f;
};

int main() 
{
	olcEngine3D demo;
	if (demo.ConstructConsole(256, 240, 2, 2))
		demo.Start();
	return 0;
}