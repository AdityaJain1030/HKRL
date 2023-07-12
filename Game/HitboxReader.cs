// ADAPTED FROM DEBUGMOD

using System.Collections.Generic;
using UnityEngine;
using GlobalEnums;
using Modding;
using UnityEngine.SceneManagement;
using HKRL.Utils;

namespace HKRL.Game
{
	public enum HitboxType : int
	{
		Knight,
		Enemy,
		Attack,
		Terrain
	}

	public class HitboxReader : MonoBehaviour
	{

		public readonly SortedDictionary<HitboxType, HashSet<Collider2D>> colliders = new()
		{
			{HitboxType.Knight, new HashSet<Collider2D>()},
			{HitboxType.Enemy, new HashSet<Collider2D>()},
			{HitboxType.Attack, new HashSet<Collider2D>()},
			{HitboxType.Terrain, new HashSet<Collider2D>()}
		};

		private void Start()
		{
			foreach (Collider2D collider2D in Resources.FindObjectsOfTypeAll<Collider2D>())
			{
				AddHitbox(collider2D);
			}
		}

		public void UpdateHitbox(GameObject go)
		{
			foreach (Collider2D col in go.GetComponentsInChildren<Collider2D>(true))
			{
				AddHitbox(col);
			}
		}

		private void AddHitbox(Collider2D collider2D)
		{
			if (collider2D == null)
			{
				return;
			}

			if (collider2D is BoxCollider2D or PolygonCollider2D or EdgeCollider2D or CircleCollider2D)
			{
				GameObject go = collider2D.gameObject;
				if (collider2D.GetComponent<DamageHero>() || collider2D.gameObject.LocateMyFSM("damages_hero"))
				{
					colliders[HitboxType.Enemy].Add(collider2D);
				}
				else if (go.layer == (int)PhysLayers.TERRAIN)
				{
					colliders[HitboxType.Terrain].Add(collider2D);
				}
				else if (go == HeroController.instance?.gameObject && !collider2D.isTrigger)
				{
					colliders[HitboxType.Knight].Add(collider2D);
				}
				else if (go.GetComponent<DamageEnemies>() || go.LocateMyFSM("damages_enemy") || go.name == "Damager" && go.LocateMyFSM("Damage"))
				{
					colliders[HitboxType.Attack].Add(collider2D);
				}
			}
		}
		public static Image RenderHitbox(Collider2D hitbox, Camera camera, byte value, Image image)
		{
			if (hitbox == null || !hitbox.isActiveAndEnabled)
			{
				return image;
			}
			switch (hitbox)
			{
				case BoxCollider2D box:
					List<Vector2> points = box.ToScreenCoordinates(camera);
					// log all points
					image.DrawPolygon(points, value);
					break;
				case CircleCollider2D circle:
					(Vector2 center, int radius) = circle.ToScreenCoordinates(camera);
					image.DrawCircle(center, radius, value);
					break;
				case PolygonCollider2D polygon:
					List<List<Vector2>> polygonPoints = polygon.ToScreenCoordinates(camera);
					foreach (var shape in polygonPoints)
					{
						image.DrawPolygon(shape, value);
					}
					break;
				case EdgeCollider2D edge:
					List<Vector2> edgePoints = edge.ToScreenCoordinates(camera);
					image.DrawPolygon(edgePoints, value);
					break;
				default:
					break;
			}
			return image;
		}
	}

	public class HitboxReaderHook
	{
		private HitboxReader _hitboxReader;
		public bool loaded = false;

		public void Load()
		{
			Unload();
			UnityEngine.SceneManagement.SceneManager.activeSceneChanged +=
				CreateHitboxReader;

			ModHooks.ColliderCreateHook += UpdateHitboxReader;

			CreateHitboxReader();
			loaded = true;
		}

		public void Unload()
		{
			UnityEngine.SceneManagement.SceneManager.activeSceneChanged -=
				CreateHitboxReader;

			ModHooks.ColliderCreateHook -= UpdateHitboxReader;
			DestroyHitboxReader();
		}

		private void CreateHitboxReader(Scene current, Scene next) =>
			CreateHitboxReader();

		private void CreateHitboxReader()
		{
			DestroyHitboxReader();
			if (GameManager.instance.IsGameplayScene())
			{
				_hitboxReader = new GameObject().AddComponent<HitboxReader>();
			}
		}

		private void DestroyHitboxReader()
		{
			if (_hitboxReader != null)
			{
				Object.Destroy(_hitboxReader);
				_hitboxReader = null;
			}
		}

		private void UpdateHitboxReader(GameObject go)
		{
			if (_hitboxReader != null)
			{
				_hitboxReader.UpdateHitbox(go);
			}
		}

		public SortedDictionary<HitboxType, HashSet<Collider2D>> GetHitboxes()
		{
			return _hitboxReader?.colliders
				?? new SortedDictionary<HitboxType, HashSet<Collider2D>>()
				{
					{ HitboxType.Knight, new HashSet<Collider2D>() },
					{ HitboxType.Enemy, new HashSet<Collider2D>() },
					{ HitboxType.Attack, new HashSet<Collider2D>() },
					{ HitboxType.Terrain, new HashSet<Collider2D>() }
				};
		}
	}

}
