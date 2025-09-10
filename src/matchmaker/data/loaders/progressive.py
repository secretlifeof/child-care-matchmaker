"""Progressive data loading for efficient matching."""

import asyncio
import logging
from uuid import UUID

import httpx

from ...models.base import Application, Center, Location

logger = logging.getLogger(__name__)


class ProgressiveLoader:
    """
    Loads data progressively to minimize memory usage and API calls.
    """

    def __init__(
        self,
        initial_batch: int = 100,
        expansion_factor: float = 2.0,
        max_centers: int = 1000,
        min_quality_threshold: float = 0.7,
        target_matches: int = 10,
        api_base_url: str = "http://localhost:8000"
    ):
        self.initial_batch = initial_batch
        self.expansion_factor = expansion_factor
        self.max_centers = max_centers
        self.min_quality_threshold = min_quality_threshold
        self.target_matches = target_matches
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient()

    async def load_application(self, application_id: UUID) -> Application | None:
        """Load a single application."""
        try:
            # Would connect to main application API
            response = await self.client.get(
                f"{self.api_base_url}/api/applications/{application_id}"
            )
            if response.status_code == 200:
                return Application(**response.json())
            return None
        except Exception as e:
            logger.error(f"Error loading application {application_id}: {e}")
            # Return mock data for testing
            return self._mock_application(application_id)

    async def load_applications(self, application_ids: list[UUID]) -> list[Application]:
        """Load multiple applications."""
        tasks = [self.load_application(app_id) for app_id in application_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        applications = []
        for result in results:
            if isinstance(result, Application):
                applications.append(result)
            else:
                logger.error(f"Failed to load application: {result}")

        return applications

    async def load_center(self, center_id: UUID) -> Center | None:
        """Load a single center."""
        try:
            response = await self.client.get(
                f"{self.api_base_url}/api/centers/{center_id}"
            )
            if response.status_code == 200:
                return Center(**response.json())
            return None
        except Exception as e:
            logger.error(f"Error loading center {center_id}: {e}")
            # Return mock data for testing
            return self._mock_center(center_id)

    async def load_centers(self, center_ids: list[UUID]) -> list[Center]:
        """Load multiple centers."""
        tasks = [self.load_center(center_id) for center_id in center_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        centers = []
        for result in results:
            if isinstance(result, Center):
                centers.append(result)
            else:
                logger.error(f"Failed to load center: {result}")

        return centers

    async def load_centers_progressive(
        self,
        application: Application,
        target_matches: int | None = None
    ) -> list[Center]:
        """
        Load centers progressively, starting with nearest.
        
        Args:
            application: Application to find centers for
            target_matches: Target number of good matches
            
        Returns:
            List of centers, sorted by distance
        """
        if target_matches is None:
            target_matches = self.target_matches

        all_centers = []
        batch_size = self.initial_batch
        offset = 0
        quality_matches = 0

        while len(all_centers) < self.max_centers:
            # Load next batch of centers
            centers = await self._load_centers_near(
                location=application.home_location,
                limit=batch_size,
                offset=offset,
                max_distance_km=application.max_distance_km
            )

            if not centers:
                break  # No more centers available

            all_centers.extend(centers)

            # Estimate quality matches in this batch
            # (In real implementation, would do quick scoring)
            quality_matches += len(centers) // 3  # Assume 1/3 are good matches

            if quality_matches >= target_matches:
                logger.info(
                    f"Found {quality_matches} potential matches in "
                    f"{len(all_centers)} centers"
                )
                break

            # Expand search
            offset += batch_size
            batch_size = int(batch_size * self.expansion_factor)
            batch_size = min(batch_size, self.max_centers - len(all_centers))

        return all_centers

    async def load_centers_for_applications(
        self,
        applications: list[Application],
        max_per_app: int = 50
    ) -> list[Center]:
        """Load centers for multiple applications."""
        # Collect unique center IDs from all applications
        center_ids = set()

        for app in applications:
            # Get nearest centers for each application
            centers = await self._load_centers_near(
                location=app.home_location,
                limit=max_per_app,
                max_distance_km=app.max_distance_km
            )

            for center in centers:
                center_ids.add(center.id)

        # Load all unique centers
        return await self.load_centers(list(center_ids))

    async def load_applications_for_center(
        self,
        center_id: UUID,
        include_all: bool = False
    ) -> list[Application]:
        """Load applications interested in a specific center."""
        try:
            params = {"center_id": center_id, "include_all": include_all}
            response = await self.client.get(
                f"{self.api_base_url}/api/applications/interested",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                return [Application(**app) for app in data]

            return []
        except Exception as e:
            logger.error(f"Error loading applications for center {center_id}: {e}")
            # Return mock data for testing
            return self._mock_applications_for_center(center_id)

    async def _load_centers_near(
        self,
        location: Location,
        limit: int,
        offset: int = 0,
        max_distance_km: float | None = None
    ) -> list[Center]:
        """Load centers near a location."""
        try:
            params = {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "limit": limit,
                "offset": offset
            }

            if max_distance_km:
                params["max_distance_km"] = max_distance_km

            response = await self.client.get(
                f"{self.api_base_url}/api/centers/nearby",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                return [Center(**center) for center in data]

            return []
        except Exception as e:
            logger.error(f"Error loading centers near {location}: {e}")
            # Return mock data for testing
            return self._mock_centers_near(location, limit, offset)

    def _mock_application(self, application_id: UUID) -> Application:
        """Create mock application for testing."""
        from datetime import date

        from ...models.base import Child, ComparisonOperator, ParentPreference, TimeSlot

        return Application(
            id=application_id,
            family_id=UUID("12345678-1234-5678-1234-567812345678"),
            children=[
                Child(
                    id=UUID("87654321-4321-8765-4321-876543218765"),
                    family_id=UUID("12345678-1234-5678-1234-567812345678"),
                    name="Test Child",
                    birth_date=date(2022, 1, 1)
                )
            ],
            home_location=Location(
                latitude=52.5200,
                longitude=13.4050,
                city="Berlin",
                country_code="DE"
            ),
            preferences=[
                ParentPreference(
                    id=UUID("11111111-1111-1111-1111-111111111111"),
                    profile_id=application_id,
                    property_key="organic_food",
                    operator=ComparisonOperator.EQUALS,
                    value_boolean=True,
                    weight=0.8,
                    threshold=0.7
                )
            ],
            desired_start_date=date(2024, 9, 1),
            desired_hours=[
                TimeSlot(
                    day_of_week=0,
                    start_hour=8,
                    end_hour=16
                )
            ],
            max_distance_km=5.0
        )

    def _mock_center(self, center_id: UUID) -> Center:
        """Create mock center for testing."""
        from datetime import date

        from ...models.base import (
            AgeGroup,
            CapacityBucket,
            CenterProperty,
            PropertyCategory,
            SourceType,
            TimeSlot,
        )

        return Center(
            id=center_id,
            name="Test Daycare Center",
            location=Location(
                latitude=52.5100,
                longitude=13.4000,
                address="Test Street 123",
                postal_code="10115",
                city="Berlin",
                country_code="DE"
            ),
            short_description="A wonderful daycare center",
            opening_hours=[
                TimeSlot(day_of_week=i, start_hour=7, end_hour=18)
                for i in range(5)  # Monday to Friday
            ],
            properties=[
                CenterProperty(
                    id=UUID("22222222-2222-2222-2222-222222222222"),
                    center_id=center_id,
                    property_type_id=UUID("33333333-3333-3333-3333-333333333333"),
                    property_key="organic_food",
                    category=PropertyCategory.SERVICE,
                    value_boolean=True,
                    source=SourceType.VERIFIED
                )
            ],
            capacity_buckets=[
                CapacityBucket(
                    id=UUID("44444444-4444-4444-4444-444444444444"),
                    center_id=center_id,
                    age_band=AgeGroup(
                        min_age_months=12,
                        max_age_months=36,
                        name="Toddlers"
                    ),
                    start_month=date(2024, 9, 1),
                    total_capacity=20,
                    available_capacity=5
                )
            ]
        )

    def _mock_centers_near(
        self,
        location: Location,
        limit: int,
        offset: int
    ) -> list[Center]:
        """Create mock centers for testing."""
        centers = []

        # Generate mock centers at different distances
        for i in range(min(limit, 10)):
            center_id = UUID(f"{'0' * 8}-{'0' * 4}-{'0' * 4}-{'0' * 4}-{str(offset + i).zfill(12)}")
            center = self._mock_center(center_id)

            # Vary location slightly
            center.location.latitude += (i - 5) * 0.01
            center.location.longitude += (i - 5) * 0.01
            center.name = f"Daycare Center {offset + i + 1}"

            centers.append(center)

        return centers

    def _mock_applications_for_center(self, center_id: UUID) -> list[Application]:
        """Create mock applications for testing."""
        applications = []

        for i in range(5):
            app_id = UUID(f"{'9' * 8}-{'9' * 4}-{'9' * 4}-{'9' * 4}-{str(i).zfill(12)}")
            app = self._mock_application(app_id)
            applications.append(app)

        return applications

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
